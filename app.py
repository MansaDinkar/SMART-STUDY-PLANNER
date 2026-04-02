"""
Smart Study Planner — Local Version
=====================================
- SQLite database (auto-created as study_planner.db)
- Local file storage (uploads/ folder)
- Groq AI for chatbot + timetable generation
- No Cloudinary, no PostgreSQL, no Render

Setup:
  1. pip install flask flask-sqlalchemy groq PyMuPDF werkzeug python-dotenv
  2. Create a .env file with: GROQ_API_KEY=gsk_...
  3. python app.py
  4. Open http://localhost:5000
"""

import os, json, re, uuid
from datetime import datetime, date, timedelta
from functools import wraps
from dotenv import load_dotenv

load_dotenv()

from flask import (Flask, render_template, request, redirect, url_for,
                   session, flash, jsonify, send_from_directory)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from groq import Groq
import cloudinary
import cloudinary.uploader
import requests as req_lib

cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key=os.environ.get("CLOUDINARY_API_KEY"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET"),
)
# ── Optional PDF text extractors ──────────────────────────────────────────────
try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except ImportError:
    _HAS_PYMUPDF = False

try:
    from pdfminer.high_level import extract_text as pdfminer_extract
    _HAS_PDFMINER = True
except ImportError:
    _HAS_PDFMINER = False

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "local-dev-secret-key-change-me")

# ── SQLite (file lives next to app.py) ────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL",
    f"sqlite:///{os.path.join(BASE_DIR, 'study_planner.db')}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

# ── Local upload folder ────────────────────────────────────────────────────────
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

db = SQLAlchemy(app)

# ── Groq client ───────────────────────────────────────────────────────────────
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

# ── Models ─────────────────────────────────────────────────────────────────────

class User(db.Model):
    __tablename__ = "users"
    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(120), nullable=False)
    email         = db.Column(db.String(180), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role          = db.Column(db.String(20), default="student")
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

    plans      = db.relationship("StudyPlan",     backref="user", lazy=True, cascade="all, delete-orphan")
    pdfs       = db.relationship("PDFNote",       backref="user", lazy=True, cascade="all, delete-orphan")
    timetables = db.relationship("SavedTimetable",backref="user", lazy=True, cascade="all, delete-orphan")


class StudyPlan(db.Model):
    __tablename__ = "study_plans"
    id            = db.Column(db.Integer, primary_key=True)
    user_id       = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    subject       = db.Column(db.String(120), nullable=False)
    study_hours   = db.Column(db.Float, nullable=False)
    schedule_date = db.Column(db.Date, nullable=False)
    completed     = db.Column(db.Boolean, default=False)
    skipped       = db.Column(db.Boolean, default=False)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)


class PDFNote(db.Model):
    __tablename__ = "pdf_notes"
    id            = db.Column(db.Integer, primary_key=True)
    user_id       = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    original_name = db.Column(db.String(260), nullable=False)
    subject       = db.Column(db.String(120))
    filename      = db.Column(db.String(260), nullable=False)  # saved filename on disk
    file_url      = db.Column(db.String(512))
    upload_date   = db.Column(db.DateTime, default=datetime.utcnow)

    chat_messages = db.relationship("ChatMessage", backref="pdf", lazy=True, cascade="all, delete-orphan")


class ChatMessage(db.Model):
    __tablename__ = "chat_messages"
    id        = db.Column(db.Integer, primary_key=True)
    pdf_id    = db.Column(db.Integer, db.ForeignKey("pdf_notes.id"), nullable=False)
    user_id   = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    role      = db.Column(db.String(10), nullable=False)
    content   = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


class SavedTimetable(db.Model):
    __tablename__ = "saved_timetables"
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    title      = db.Column(db.String(260))
    data       = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# ── Create tables ─────────────────────────────────────────────────────────────
with app.app_context():
    db.create_all()

# ── Auth helpers ──────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in first.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("user_role") != "admin":
            flash("Admin access required.", "danger")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return decorated


# ── PDF text extraction ───────────────────────────────────────────────────────

def extract_pdf_text_from_path(filepath: str) -> str:
    if _HAS_PYMUPDF:
        try:
            doc = fitz.open(filepath)
            text = "\n".join(page.get_text() for page in doc).strip()
            doc.close()
            if len(text) > 100:
                return text
        except Exception:
            pass

    if _HAS_PDFMINER:
        try:
            text = pdfminer_extract(filepath)
            if text and len(text.strip()) > 100:
                return text.strip()
        except Exception:
            pass

    return ""


# ── Groq AI helper ────────────────────────────────────────────────────────────

PRIMARY_MODEL  = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"

def groq_chat(messages, model=None, temperature=0.4, max_tokens=4096):
    model = model or PRIMARY_MODEL
    try:
        resp = groq_client.chat.completions.create(
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
    except Exception as e:
        err = str(e)
        if ("rate_limit" in err or "429" in err) and model == PRIMARY_MODEL:
            resp = groq_client.chat.completions.create(
                model=FALLBACK_MODEL, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        raise


# ════════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name     = request.form.get("name", "").strip()
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm_password", "")
        role     = request.form.get("role", "student")

        if not name or not email or not password:
            flash("All fields are required.", "danger")
            return render_template("register.html")
        if password != confirm:
            flash("Passwords do not match.", "danger")
            return render_template("register.html")
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return render_template("register.html")
        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "danger")
            return render_template("register.html")

        user = User(
            name=name, email=email,
            password_hash=generate_password_hash(password),
            role=role if role in ("student", "admin") else "student",
        )
        db.session.add(user)
        db.session.commit()
        flash("Account created! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user     = User.query.filter_by(email=email).first()

        if not user or not check_password_hash(user.password_hash, password):
            flash("Invalid email or password.", "danger")
            return render_template("login.html")

        session["user_id"]   = user.id
        session["user_name"] = user.name
        session["user_role"] = user.role
        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


# ── Dashboard ─────────────────────────────────────────────────────────────────

@app.route("/dashboard")
@login_required
def dashboard():
    uid   = session["user_id"]
    today = date.today()
    now   = datetime.now()

    plans_all = StudyPlan.query.filter_by(user_id=uid).all()

    today_plans = [p for p in plans_all if p.schedule_date == today]
    today_total = len(today_plans)
    today_done  = sum(1 for p in today_plans if p.completed)
    today_pct   = round(today_done / today_total * 100) if today_total else 0

    streak = 0
    # Include today in streak if all today's plans are done
    if today_plans and all(p.completed for p in today_plans):
        streak += 1
        check = today - timedelta(days=1)
    else:
        check = today - timedelta(days=1)
    while True:
        day_plans = [p for p in plans_all if p.schedule_date == check]
        if day_plans and all(p.completed for p in day_plans):
            streak += 1
            check  -= timedelta(days=1)
        else:
            break

    completed_plans = [p for p in plans_all if p.completed]
    skipped_plans   = [p for p in plans_all if p.skipped]
    pending_plans   = [p for p in plans_all if not p.completed and not p.skipped]
    hours_studied   = sum(p.study_hours for p in completed_plans)
    total_hours     = sum(p.study_hours for p in plans_all)

    stats = {
        "today_pct":       today_pct,
        "today_completed": today_done,
        "today_total":     today_total,
        "streak":          streak,
        "hours_studied":   round(hours_studied, 1),
        "total_hours":     round(total_hours, 1),
        "completed":       len(completed_plans),
        "skipped":         len(skipped_plans),
        "pending":         len(pending_plans),
    }

    upcoming = (
        StudyPlan.query
        .filter_by(user_id=uid, skipped=False, completed=False)
        .order_by(StudyPlan.schedule_date)
        .limit(20).all()
    )

    skipped_sessions = (
        StudyPlan.query
        .filter_by(user_id=uid, skipped=True)
        .order_by(StudyPlan.schedule_date.desc())
        .limit(10).all()
    )

    cal_year  = today.year
    cal_month = today.month
    import calendar as cal_mod
    _, days_in_month = cal_mod.monthrange(cal_year, cal_month)
    cal_data = {}
    for d in range(1, days_in_month + 1):
        day_plans = [p for p in plans_all if p.schedule_date == date(cal_year, cal_month, d)]
        if day_plans:
            cal_data[d] = {
                "total":   len(day_plans),
                "done":    sum(1 for p in day_plans if p.completed),
                "skipped": sum(1 for p in day_plans if p.skipped),
                "pending": sum(1 for p in day_plans if not p.completed and not p.skipped),
            }

    pdfs = PDFNote.query.filter_by(user_id=uid).order_by(PDFNote.upload_date.desc()).limit(5).all()

    return render_template(
        "dashboard.html",
        stats=stats, plans=upcoming, skipped_sessions=skipped_sessions,
        pdfs=pdfs, now=now, today=today,
        cal_data=cal_data, cal_year=cal_year, cal_month=cal_month,
    )


# ── Study Plans ───────────────────────────────────────────────────────────────

@app.route("/study-plans")
@login_required
def study_plans():
    return render_template("study_plans.html")


@app.route("/add-plan", methods=["POST"])
@login_required
def add_plan():
    subject = request.form.get("subject", "").strip()
    hours   = request.form.get("study_hours", "")
    sdate   = request.form.get("schedule_date", "")
    try:
        hours = float(hours)
        sdate = date.fromisoformat(sdate)
    except (ValueError, TypeError):
        flash("Invalid input.", "danger")
        return redirect(url_for("dashboard"))

    plan = StudyPlan(user_id=session["user_id"], subject=subject,
                     study_hours=hours, schedule_date=sdate)
    db.session.add(plan)
    db.session.commit()
    flash(f"Added '{subject}' on {sdate.strftime('%b %d')}.", "success")
    return redirect(url_for("dashboard"))


@app.route("/complete-plan/<int:plan_id>", methods=["POST"])
@login_required
def complete_plan(plan_id):
    plan = StudyPlan.query.get_or_404(plan_id)
    if plan.user_id != session["user_id"]:
        return "Forbidden", 403
    plan.completed = True
    plan.skipped   = False
    db.session.commit()
    return redirect(request.referrer or url_for("dashboard"))


@app.route("/skip-plan/<int:plan_id>", methods=["POST"])
@login_required
def skip_and_redistribute(plan_id):
    plan = StudyPlan.query.get_or_404(plan_id)
    if plan.user_id != session["user_id"]:
        return "Forbidden", 403
    plan.skipped = True
    next_plan = (
        StudyPlan.query
        .filter(
            StudyPlan.user_id   == session["user_id"],
            StudyPlan.subject   == plan.subject,
            StudyPlan.skipped   == False,
            StudyPlan.completed == False,
            StudyPlan.schedule_date > plan.schedule_date,
        )
        .order_by(StudyPlan.schedule_date).first()
    )
    if next_plan:
        next_plan.study_hours = round(next_plan.study_hours + plan.study_hours, 1)
    db.session.commit()
    return redirect(request.referrer or url_for("dashboard"))


@app.route("/unskip-plan/<int:plan_id>", methods=["POST"])
@login_required
def unskip_plan(plan_id):
    plan = StudyPlan.query.get_or_404(plan_id)
    if plan.user_id != session["user_id"]:
        return "Forbidden", 403
    plan.skipped = False
    db.session.commit()
    return redirect(request.referrer or url_for("dashboard"))


@app.route("/delete-all-plans", methods=["POST"])
@login_required
def delete_all_plans():
    StudyPlan.query.filter_by(user_id=session["user_id"]).delete()
    db.session.commit()
    flash("All study plans deleted.", "success")
    return redirect(url_for("study_plans"))


# ── AI endpoints ──────────────────────────────────────────────────────────────

@app.route("/study-plans/extract-syllabus", methods=["POST"])
@login_required
def extract_syllabus():
    f = request.files.get("syllabus_file")
    if not f:
        return jsonify({"error": "No file uploaded."}), 400

    # Save temporarily to extract text
    tmp_path = os.path.join(UPLOAD_FOLDER, f"tmp_{uuid.uuid4().hex}.pdf")
    f.save(tmp_path)
    text = extract_pdf_text_from_path(tmp_path)
    os.remove(tmp_path)

    if not text or len(text) < 80:
        return jsonify({"error": "Could not extract text from this PDF. Make sure it's a digital (not scanned) PDF."}), 400

    prompt = f"""You are an academic syllabus analyser.

Given the following syllabus text, extract ALL modules/units/subjects and their topics.

Return ONLY valid JSON — no markdown, no explanation, no code blocks — in this exact structure:
{{
  "subjects": [
    {{
      "subject": "Module or Subject Name",
      "priority": "high",
      "topics": ["Topic A", "Topic B", "Topic C"]
    }}
  ]
}}

Rules:
- priority must be one of: high, medium, low
- Extract every unit/module as a separate subject
- Each topic should be a short phrase (3-8 words)
- Return at minimum 3 topics per subject

SYLLABUS TEXT:
{text[:12000]}
"""
    try:
        raw  = groq_chat([{"role": "user", "content": prompt}], max_tokens=3000)
        raw  = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
        data = json.loads(raw)
        return jsonify(data)
    except json.JSONDecodeError:
        return jsonify({"error": "AI returned invalid JSON. Try again."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/study-plans/generate-timetable", methods=["POST"])
@login_required
def generate_timetable():
    body          = request.get_json(force=True)
    subjects      = body.get("subjects", [])
    start_date    = body.get("start_date", "")
    exam_date     = body.get("exam_date", "")
    num_days      = int(body.get("num_days", 7))
    hours_per_day = float(body.get("hours_per_day", 6))
    start_time    = body.get("start_time", "08:00")
    break_days    = int(body.get("break_days", 1))
    break_minutes = body.get("break_minutes", "pomodoro")
    extra         = body.get("extra_instructions", "")

    if not subjects:
        return jsonify({"error": "No subjects provided."}), 400

    prompt = f"""You are an expert academic study planner.

Create a detailed, realistic day-by-day study timetable.

SUBJECTS & TOPICS:
{json.dumps(subjects, indent=2)}

SETTINGS:
- Start date: {start_date}
- Exam/End date: {exam_date}
- Total study days: {num_days}
- Study hours per day: {hours_per_day}
- Study starts at: {start_time}
- Rest days per week: {break_days}
- Break style: {break_minutes}
- Extra instructions: {extra or "None"}

Return ONLY valid JSON (no markdown, no explanation) with this EXACT structure:
{{
  "startDate": "YYYY-MM-DD",
  "examDate": "YYYY-MM-DD",
  "totalStudyDays": <number>,
  "totalHours": <number>,
  "subjects": ["Subject1", "Subject2"],
  "days": [
    {{
      "date": "YYYY-MM-DD",
      "dayName": "Monday",
      "type": "study",
      "totalHours": <number>,
      "sessions": [
        {{
          "subject": "Subject Name",
          "durationHours": <number>,
          "topics": ["Topic 1", "Topic 2"],
          "note": "Optional tip"
        }}
      ]
    }}
  ]
}}
"""
    try:
        raw  = groq_chat([{"role": "user", "content": prompt}], max_tokens=4096, temperature=0.3)
        raw  = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
        data = json.loads(raw)
        return jsonify(data)
    except json.JSONDecodeError:
        return jsonify({"error": "AI returned invalid JSON. Try regenerating."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/study-plans/ai-generate/check-conflicts", methods=["POST"])
@login_required
def check_conflicts():
    body      = request.get_json(force=True)
    dates     = body.get("dates", [])
    day_hours = float(body.get("day_hours", 6))
    uid       = session["user_id"]

    conflicts = []
    fully_booked = []
    existing_subjects = set()

    for d in dates:
        try:
            d_obj = date.fromisoformat(d)
        except Exception:
            continue
        existing = StudyPlan.query.filter_by(user_id=uid, schedule_date=d_obj, skipped=False, completed=False).all()
        if existing:
            conflicts.append(str(d_obj))
            hours_used = sum(p.study_hours for p in existing)
            for p in existing:
                existing_subjects.add(p.subject)
            if hours_used >= day_hours:
                fully_booked.append(str(d_obj))

    return jsonify({"conflicts": conflicts, "fully_booked": fully_booked, "existing_subjects": list(existing_subjects)})


@app.route("/study-plans/ai-generate/save", methods=["POST"])
@login_required
def ai_save_plan():
    body     = request.get_json(force=True)
    schedule = body.get("schedule", [])
    mode     = body.get("mode", "merge")
    day_hours= float(body.get("day_hours", 6))
    uid      = session["user_id"]

    if not schedule:
        return jsonify({"success": False, "error": "Empty schedule."}), 400

    saved = 0
    for entry in schedule:
        try:
            sdate = date.fromisoformat(entry["date"])
            subj  = entry.get("module") or entry.get("subject", "")
            hrs   = float(entry.get("hours", 1))
        except Exception:
            continue

        if mode == "replace":
            StudyPlan.query.filter_by(user_id=uid, schedule_date=sdate, completed=False).delete()
        elif mode == "merge":
            existing_hours = db.session.query(
                db.func.sum(StudyPlan.study_hours)
            ).filter_by(user_id=uid, schedule_date=sdate, skipped=False).scalar() or 0
            remaining = day_hours - existing_hours
            if remaining <= 0:
                continue
            hrs = min(hrs, remaining)

        plan = StudyPlan(user_id=uid, subject=subj, study_hours=round(hrs, 1), schedule_date=sdate)
        db.session.add(plan)
        saved += 1

    db.session.commit()
    return jsonify({"success": True, "saved": saved})


# ── Saved Timetables ──────────────────────────────────────────────────────────

@app.route("/study-plans/save-timetable", methods=["POST"])
@login_required
def save_timetable():
    plan  = request.get_json(force=True)
    uid   = session["user_id"]
    title = plan.get("title", f"Timetable {datetime.now().strftime('%b %d %Y')}")
    tt    = SavedTimetable(user_id=uid, title=title, data=json.dumps(plan))
    db.session.add(tt)
    db.session.commit()
    return jsonify({"success": True, "id": tt.id})


@app.route("/study-plans/load-timetable", methods=["GET"])
@login_required
def load_timetable():
    uid  = session["user_id"]
    rows = SavedTimetable.query.filter_by(user_id=uid).order_by(SavedTimetable.created_at.desc()).all()
    timetables = []
    for row in rows:
        try:
            timetables.append({
                "id":         row.id,
                "title":      row.title,
                "created_at": row.created_at.strftime("%b %d, %Y"),
                "timetable":  json.loads(row.data),
            })
        except Exception:
            pass
    latest = timetables[0]["timetable"] if timetables else None
    return jsonify({"timetable": latest, "timetables": timetables})


@app.route("/study-plans/delete-timetable", methods=["POST"])
@login_required
def delete_timetable():
    uid  = session["user_id"]
    body = request.get_json(force=True) or {}
    tid  = body.get("id")
    if tid:
        tt = SavedTimetable.query.filter_by(id=tid, user_id=uid).first()
        if tt:
            db.session.delete(tt)
            db.session.commit()
    else:
        SavedTimetable.query.filter_by(user_id=uid).delete()
        db.session.commit()
    return jsonify({"success": True})


@app.route("/study-plans/finish-session", methods=["POST"])
@login_required
def finish_session():
    return jsonify({"success": True})


@app.route("/my-timetable")
@login_required
def my_timetable():
    return render_template("my_timetable.html")


# ── PDF Notes ─────────────────────────────────────────────────────────────────

@app.route("/pdf-notes")
@login_required
def pdf_notes():
    uid  = session["user_id"]
    pdfs = PDFNote.query.filter_by(user_id=uid).order_by(PDFNote.upload_date.desc()).all()
    return render_template("pdf_notes.html", pdfs=pdfs)


@app.route("/upload-pdf", methods=["POST"])
@login_required
def upload_pdf():
    f       = request.files.get("pdf_file")
    subject = request.form.get("subject", "").strip()

    if not f or not f.filename.lower().endswith(".pdf"):
        flash("Please upload a valid PDF file.", "danger")
        return redirect(url_for("pdf_notes"))

    result = cloudinary.uploader.upload(
        f,
        resource_type="raw",
        folder="study_planner_pdfs",
        public_id=f"{session['user_id']}_{uuid.uuid4().hex}",
    )

    note = PDFNote(
        user_id       = session["user_id"],
        original_name = f.filename,
        subject       = subject or None,
        filename      = result["public_id"],
        file_url      = result["secure_url"],
    )
    db.session.add(note)
    db.session.commit()
    flash(f"'{f.filename}' uploaded successfully.", "success")
    return redirect(url_for("pdf_notes"))

@app.route("/delete-pdf/<int:pdf_id>", methods=["POST"])
@login_required
def delete_pdf(pdf_id):
    pdf = PDFNote.query.get_or_404(pdf_id)
    if pdf.user_id != session["user_id"]:
        return "Forbidden", 403

    try:
        cloudinary.uploader.destroy(pdf.filename, resource_type="raw")
    except Exception:
        pass

    db.session.delete(pdf)
    db.session.commit()
    flash("PDF deleted.", "success")
    return redirect(url_for("pdf_notes"))


# ── PDF Chatbot ───────────────────────────────────────────────────────────────

@app.route("/chatbot")
@login_required
def chatbot():
    uid      = session["user_id"]
    pdfs     = PDFNote.query.filter_by(user_id=uid).order_by(PDFNote.upload_date.desc()).all()
    pdf_id   = request.args.get("pdf_id", type=int)
    selected = None
    history  = []

    if pdf_id:
        selected = PDFNote.query.filter_by(id=pdf_id, user_id=uid).first()
        if selected:
            msgs = (
                ChatMessage.query
                .filter_by(pdf_id=pdf_id, user_id=uid)
                .order_by(ChatMessage.timestamp).all()
            )
            history = [{"role": m.role, "content": m.content} for m in msgs]

    return render_template("chatbot.html", pdfs=pdfs, selected_pdf=selected, chat_history=history)


@app.route("/chatbot/query", methods=["POST"])
@login_required
def chatbot_query():
    uid    = session["user_id"]
    body   = request.get_json(force=True)
    query  = body.get("query", "").strip()
    pdf_id = body.get("pdf_id")

    if not query or not pdf_id:
        return jsonify({"error": "Missing query or pdf_id."}), 400

    pdf = PDFNote.query.filter_by(id=pdf_id, user_id=uid).first()
    if not pdf:
        return jsonify({"error": "PDF not found."}), 404

    # Handle greetings instantly without touching the PDF
    greetings = {"hi", "hey", "hello", "hiya", "howdy", "sup", "yo", "hi there", "hey there", "hello there"}
    if query.lower().strip() in greetings:
        answer = f"Hey! 👋 I'm your study assistant for **{pdf.original_name}**. Ask me anything about this document — concepts, definitions, summaries, or specific topics!"
        db.session.add(ChatMessage(pdf_id=pdf_id, user_id=uid, role="user", content=query))
        db.session.add(ChatMessage(pdf_id=pdf_id, user_id=uid, role="assistant", content=answer))
        db.session.commit()
        return jsonify({"answer": answer})

    import tempfile
    pdf_url = pdf.file_url
    if not pdf_url:
        return jsonify({"error": "PDF URL missing."}), 404

    try:
        r = req_lib.get(pdf_url, timeout=15)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(r.content)
            tmp_path = tmp.name
        pdf_text = extract_pdf_text_from_path(tmp_path)
        os.remove(tmp_path)
    except Exception as e:
        return jsonify({"error": f"Could not fetch PDF: {e}"}), 500

    if not pdf_text or len(pdf_text) < 50:
        return jsonify({"answer": "I couldn't read text from this PDF. It may be a scanned image PDF."})

    recent = (
        ChatMessage.query
        .filter_by(pdf_id=pdf_id, user_id=uid)
        .order_by(ChatMessage.timestamp.desc())
        .limit(10).all()
    )
    recent.reverse()
    history_msgs = [{"role": m.role, "content": m.content} for m in recent]

    system_prompt = f"""You are a smart, friendly study assistant — like ChatGPT but focused on helping students understand their study material.

You have been given the content of a PDF document called "{pdf.original_name}". Your job is to help the student understand it.

HOW TO BEHAVE:
- Answer questions in a clear, natural, conversational way — like a knowledgeable friend explaining things
- When explaining concepts, use simple language and examples where helpful
- If asked to summarize, give a well-structured summary in your own words
- If asked to define terms, give clear definitions with context from the document
- If a question is not covered in the PDF, say so honestly and offer what you do know
- Never paste raw text, tables of contents, or numbered chapter lists directly — always explain and rephrase
- Keep responses concise but complete — don't ramble
- Use bullet points or numbered lists only when it genuinely helps clarity
- Be encouraging and student-friendly

PDF CONTENT:
{pdf_text[:10000]}
"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history_msgs)
    messages.append({"role": "user", "content": query})

    try:
        answer = groq_chat(messages, max_tokens=1500, temperature=0.5)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    db.session.add(ChatMessage(pdf_id=pdf_id, user_id=uid, role="user",      content=query))
    db.session.add(ChatMessage(pdf_id=pdf_id, user_id=uid, role="assistant", content=answer))
    db.session.commit()

    return jsonify({"answer": answer})


@app.route("/chatbot/clear", methods=["POST"])
@login_required
def clear_chat():
    uid    = session["user_id"]
    body   = request.get_json(force=True)
    pdf_id = body.get("pdf_id")
    if pdf_id:
        ChatMessage.query.filter_by(pdf_id=pdf_id, user_id=uid).delete()
        db.session.commit()
    return jsonify({"success": True})


# ── Admin ─────────────────────────────────────────────────────────────────────

@app.route("/admin")
@login_required
@admin_required
def admin_panel():
    users = User.query.order_by(User.created_at.desc()).all()
    return render_template("admin_panel.html", users=users)


@app.route("/admin/delete-user/<int:uid>", methods=["POST"])
@login_required
@admin_required
def admin_delete_user(uid):
    if uid == session["user_id"]:
        flash("You can't delete yourself.", "danger")
        return redirect(url_for("admin_panel"))
    user = User.query.get_or_404(uid)
    for pdf in user.pdfs:
        filepath = os.path.join(UPLOAD_FOLDER, pdf.filename)
        if os.path.exists(filepath):
            os.remove(filepath)
    db.session.delete(user)
    db.session.commit()
    flash(f"User '{user.name}' deleted.", "success")
    return redirect(url_for("admin_panel"))


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)
