from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime, date, timedelta
from functools import wraps
from collections import defaultdict
import os

from pathlib import Path
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                os.environ.setdefault(key.strip(), val.strip())

try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_SUPPORT = True
except ImportError:
    OCR_SUPPORT = False

GEMINI_SUPPORT = False
gemini_model   = None

try:
    from groq import Groq
    GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
    if GROQ_KEY:
        groq_client    = Groq(api_key=GROQ_KEY)
        GEMINI_SUPPORT = True
except Exception as e:
    print(f"[Groq] Setup error: {e}")

app = Flask(__name__, template_folder='TEMPLATES')
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///study_planner.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

ALLOWED_EXTENSIONS = {'pdf'}
db = SQLAlchemy(app)


class User(db.Model):
    __tablename__ = 'users'
    id       = db.Column(db.Integer, primary_key=True)
    name     = db.Column(db.String(100), nullable=False)
    email    = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)
    role     = db.Column(db.String(20), default='student')
    plans    = db.relationship('StudyPlan', backref='user', lazy=True, cascade='all, delete')
    pdfs     = db.relationship('PDFNote',   backref='user', lazy=True, cascade='all, delete')

class StudyPlan(db.Model):
    __tablename__ = 'study_plans'
    id            = db.Column(db.Integer, primary_key=True)
    user_id       = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    subject       = db.Column(db.String(150), nullable=False)
    study_hours   = db.Column(db.Float, nullable=False)
    schedule_date = db.Column(db.Date, nullable=False)
    completed     = db.Column(db.Boolean, default=False)
    skipped       = db.Column(db.Boolean, default=False)   # ← NEW
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

class PDFNote(db.Model):
    __tablename__ = 'pdf_notes'
    id            = db.Column(db.Integer, primary_key=True)
    user_id       = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    filename      = db.Column(db.String(255), nullable=False)
    original_name = db.Column(db.String(255), nullable=False)
    subject       = db.Column(db.String(150))
    upload_date   = db.Column(db.DateTime, default=datetime.utcnow)
    text_content  = db.Column(db.Text)

class SyllabusFile(db.Model):
    __tablename__ = 'syllabus_files'
    id            = db.Column(db.Integer, primary_key=True)
    user_id       = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    filename      = db.Column(db.String(255), nullable=False)
    original_name = db.Column(db.String(255), nullable=False)
    module_name   = db.Column(db.String(150))
    upload_date   = db.Column(db.DateTime, default=datetime.utcnow)
    text_content  = db.Column(db.Text)

class SavedTimetable(db.Model):
    __tablename__ = 'saved_timetables'
    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title        = db.Column(db.String(255), nullable=False)
    timetable_json = db.Column(db.Text, nullable=False)   # full JSON blob
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at   = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_pdf_text(filepath):
    text = ''
    if PDF_SUPPORT:
        try:
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'
        except Exception as e:
            print(f"[PyPDF2] Error: {e}")
    if not text.strip() and OCR_SUPPORT:
        try:
            images = convert_from_path(filepath, dpi=200)
            for image in images:
                page_text = pytesseract.image_to_string(image)
                if page_text:
                    text += page_text + '\n'
        except Exception as e:
            print(f"[OCR] Error: {e}")
    return text.strip()


def ask_gemini(pdf_text, question, chat_history):
    if not GEMINI_SUPPORT:
        return "AI is not configured. Please check your GROQ_API_KEY in the .env file."
    pdf_excerpt = pdf_text[:6000] if pdf_text and len(pdf_text) > 6000 else (pdf_text or '')
    doc_context = (f"\n\nReference document:\n{pdf_excerpt}" if pdf_excerpt else "")
    messages = [{"role": "system", "content": (
        "You are a brilliant AI study assistant.\n"
        "EXAM RULES: 2 marks=80 words, 4 marks=150 words, 7 marks=350 words essay, 10 marks=500 words.\n"
        "Essays use paragraphs NOT bullets. Use **bold** for key terms." + doc_context
    )}]
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile", messages=messages, max_tokens=1500, temperature=0.7)
        return response.choices[0].message.content
    except Exception as e:
        return f"AI error: {str(e)}"


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to continue.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        user = User.query.get(session['user_id'])
        if not user or user.role != 'admin':
            flash('Admin access required.', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated


@app.context_processor
def inject_now():
    return {'now': datetime.utcnow()}


@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name     = request.form.get('name', '').strip()
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm_password', '')
        role     = request.form.get('role', 'student')
        if not all([name, email, password, confirm]):
            flash('All fields are required.', 'danger')
        elif password != confirm:
            flash('Passwords do not match.', 'danger')
        elif len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
        elif User.query.filter_by(email=email).first():
            flash('Email already registered.', 'danger')
        else:
            user = User(name=name, email=email, password=generate_password_hash(password), role=role)
            db.session.add(user)
            db.session.commit()
            flash('Account created! Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        user     = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id']   = user.id
            session['user_name'] = user.name
            session['user_role'] = user.role
            flash(f'Welcome back, {user.name}!', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid email or password.', 'danger')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


@app.route('/dashboard')
@login_required
def dashboard():
    uid   = session['user_id']
    today = date.today()

    # ── Auto-skip past incomplete plans ──────────────────────────────────────
    past_incomplete = StudyPlan.query.filter(
        StudyPlan.user_id == uid,
        StudyPlan.completed == False,
        StudyPlan.skipped == False,
        StudyPlan.schedule_date < today
    ).all()
    for p in past_incomplete:
        p.skipped = True
    if past_incomplete:
        db.session.commit()

    # ── Upcoming plans (today + future, not skipped) ──────────────────────────
    plans = StudyPlan.query.filter(
        StudyPlan.user_id == uid,
        StudyPlan.schedule_date >= today,
        StudyPlan.skipped == False
    ).order_by(StudyPlan.schedule_date).limit(10).all()

    pdfs = PDFNote.query.filter_by(user_id=uid).order_by(PDFNote.upload_date.desc()).limit(5).all()

    # ── Core stats ────────────────────────────────────────────────────────────
    all_plans       = StudyPlan.query.filter_by(user_id=uid).all()
    total_plans     = len(all_plans)
    completed_plans = sum(1 for p in all_plans if p.completed)
    skipped_plans   = sum(1 for p in all_plans if p.skipped)
    total_pdfs      = PDFNote.query.filter_by(user_id=uid).count()

    hours_planned  = sum(p.study_hours for p in all_plans)
    hours_studied  = sum(p.study_hours for p in all_plans if p.completed)

    # ── Today's progress ──────────────────────────────────────────────────────
    today_plans     = [p for p in all_plans if p.schedule_date == today]
    today_total     = len(today_plans)
    today_completed = sum(1 for p in today_plans if p.completed)
    today_pct       = round((today_completed / today_total * 100) if today_total else 0)

    # ── Streak (consecutive days with ≥1 completed plan) ─────────────────────
    completed_dates = sorted({p.schedule_date for p in all_plans if p.completed}, reverse=True)
    streak = 0
    check  = today
    for d in completed_dates:
        if d == check:
            streak += 1
            check  -= timedelta(days=1)
        elif d < check:
            break

    # ── 7-week heatmap (49 days back from today) ──────────────────────────────
    heatmap_start = today - timedelta(days=48)
    heatmap = {}
    for p in all_plans:
        if heatmap_start <= p.schedule_date <= today:
            ds = p.schedule_date.isoformat()
            if ds not in heatmap:
                heatmap[ds] = {'total': 0, 'done': 0, 'skipped': 0}
            heatmap[ds]['total'] += 1
            if p.completed:
                heatmap[ds]['done'] += 1
            elif p.skipped:
                heatmap[ds]['skipped'] += 1

    # ── Most-missed subjects ──────────────────────────────────────────────────
    miss_count = defaultdict(int)
    for p in all_plans:
        if p.skipped:
            miss_count[p.subject] += 1
    most_missed = sorted(miss_count.items(), key=lambda x: -x[1])[:5]
    max_missed  = most_missed[0][1] if most_missed else 1

    # ── Calendar data (current month) ─────────────────────────────────────────
    cal_year  = today.year
    cal_month = today.month
    import calendar
    cal_days_count = calendar.monthrange(cal_year, cal_month)[1]
    cal_data   = {}
    for p in all_plans:
        if p.schedule_date.year == cal_year and p.schedule_date.month == cal_month:
            d = p.schedule_date.day
            if d not in cal_data:
                cal_data[d] = {'total': 0, 'done': 0, 'skipped': 0, 'pending': 0}
            cal_data[d]['total'] += 1
            if p.completed:
                cal_data[d]['done'] += 1
            elif p.skipped:
                cal_data[d]['skipped'] += 1
            else:
                cal_data[d]['pending'] += 1
    cal_first_weekday = calendar.monthrange(cal_year, cal_month)[0]  # 0=Mon

    stats = {
        'total_plans':    total_plans,
        'completed':      completed_plans,
        'pending':        total_plans - completed_plans - skipped_plans,
        'skipped':        skipped_plans,
        'total_pdfs':     total_pdfs,
        'total_hours':    round(hours_planned, 1),
        'hours_studied':  round(hours_studied, 1),
        'today_pct':      today_pct,
        'today_total':    today_total,
        'today_completed':today_completed,
        'streak':         streak,
    }

    # ── Skipped sessions (for the restore box) ───────────────────────────────
    skipped_sessions = StudyPlan.query.filter(
        StudyPlan.user_id == uid,
        StudyPlan.skipped == True
    ).order_by(StudyPlan.schedule_date.desc()).all()

    return render_template('dashboard.html',
        plans=plans, pdfs=pdfs, stats=stats,
        heatmap=heatmap, heatmap_start=heatmap_start.isoformat(),
        most_missed=most_missed, max_missed=max_missed,
        cal_data=cal_data, cal_year=cal_year, cal_month=cal_month,
        cal_days_count=cal_days_count, cal_first_weekday=cal_first_weekday,
        today=today, skipped_sessions=skipped_sessions
    )


@app.route('/study-plans')
@login_required
def study_plans():
    uid   = session['user_id']
    plans = StudyPlan.query.filter_by(user_id=uid).order_by(StudyPlan.schedule_date).all()
    pdfs  = PDFNote.query.filter_by(user_id=uid).order_by(PDFNote.upload_date.desc()).all()
    return render_template('study_plans.html', plans=plans, pdfs=pdfs)


@app.route('/study-plans/add', methods=['POST'])
@login_required
def add_plan():
    subject       = request.form.get('subject', '').strip()
    study_hours   = request.form.get('study_hours', '')
    schedule_date = request.form.get('schedule_date', '')
    if not all([subject, study_hours, schedule_date]):
        flash('All fields are required.', 'danger')
        return redirect(request.referrer or url_for('study_plans'))
    try:
        plan = StudyPlan(user_id=session['user_id'], subject=subject,
                         study_hours=float(study_hours),
                         schedule_date=datetime.strptime(schedule_date, '%Y-%m-%d').date())
        db.session.add(plan)
        db.session.commit()
        flash(f'Study plan for "{subject}" added!', 'success')
    except ValueError:
        flash('Invalid date or hours.', 'danger')
    return redirect(request.referrer or url_for('study_plans'))


@app.route('/study-plans/<int:plan_id>/complete', methods=['POST'])
@login_required
def complete_plan(plan_id):
    plan = StudyPlan.query.filter_by(id=plan_id, user_id=session['user_id']).first_or_404()
    plan.completed = True
    plan.skipped   = False
    db.session.commit()
    flash('Plan marked as completed!', 'success')
    return redirect(request.referrer or url_for('study_plans'))


@app.route('/study-plans/<int:plan_id>/uncomplete', methods=['POST'])
@login_required
def uncomplete_plan(plan_id):
    plan = StudyPlan.query.filter_by(id=plan_id, user_id=session['user_id']).first_or_404()
    plan.completed = False
    db.session.commit()
    flash('Plan marked as not done.', 'info')
    return redirect(request.referrer or url_for('study_plans'))


@app.route('/study-plans/<int:plan_id>/skip', methods=['POST'])
@login_required
def skip_plan(plan_id):
    plan = StudyPlan.query.filter_by(id=plan_id, user_id=session['user_id']).first_or_404()
    plan.skipped   = True
    plan.completed = False
    db.session.commit()
    flash(f'"{plan.subject}" marked as skipped.', 'warning')
    return redirect(request.referrer or url_for('dashboard'))


@app.route('/study-plans/<int:plan_id>/skip-redistribute', methods=['POST'])
@login_required
def skip_and_redistribute(plan_id):
    """
    Skip a session and redistribute its hours evenly across future pending sessions
    of the same subject.
    """
    uid  = session['user_id']
    plan = StudyPlan.query.filter_by(id=plan_id, user_id=uid).first_or_404()

    skipped_subject = plan.subject
    skipped_hours   = plan.study_hours
    skipped_date    = plan.schedule_date

    # Mark this session as skipped
    plan.skipped   = True
    plan.completed = False

    # Find all future pending sessions of the same subject
    today = date.today()
    future_sessions = StudyPlan.query.filter(
        StudyPlan.user_id    == uid,
        StudyPlan.subject    == skipped_subject,
        StudyPlan.skipped    == False,
        StudyPlan.completed  == False,
        StudyPlan.schedule_date > skipped_date
    ).order_by(StudyPlan.schedule_date).all()

    redistributed = 0
    if future_sessions and skipped_hours > 0:
        # Spread the skipped hours evenly across future sessions
        extra_per_session = round(skipped_hours / len(future_sessions), 1)
        for fs in future_sessions:
            fs.study_hours = round(fs.study_hours + extra_per_session, 1)
            redistributed += 1

    db.session.commit()

    if redistributed:
        flash(
            f'"{skipped_subject}" skipped. {skipped_hours}h redistributed across '
            f'{redistributed} upcoming session(s) (+{extra_per_session}h each).',
            'warning'
        )
    else:
        flash(
            f'"{skipped_subject}" skipped. No future sessions found to redistribute to.',
            'warning'
        )

    return redirect(request.referrer or url_for('dashboard'))


@app.route('/study-plans/<int:plan_id>/unskip', methods=['POST'])
@login_required
def unskip_plan(plan_id):
    plan = StudyPlan.query.filter_by(id=plan_id, user_id=session['user_id']).first_or_404()
    plan.skipped = False
    db.session.commit()
    flash(f'"{plan.subject}" restored.', 'success')
    return redirect(request.referrer or url_for('dashboard'))


@app.route('/study-plans/<int:plan_id>/delete', methods=['POST'])
@login_required
def delete_plan(plan_id):
    plan = StudyPlan.query.filter_by(id=plan_id, user_id=session['user_id']).first_or_404()
    db.session.delete(plan)
    db.session.commit()
    flash('Plan deleted.', 'info')
    return redirect(request.referrer or url_for('study_plans'))


@app.route('/study-plans/delete-all', methods=['POST'])
@login_required
def delete_all_plans():
    uid = session['user_id']
    StudyPlan.query.filter_by(user_id=uid).delete()
    SavedTimetable.query.filter_by(user_id=uid).delete()
    db.session.commit()
    flash('All study plans and timetables deleted.', 'info')
    return redirect(url_for('study_plans'))


@app.route('/syllabus/upload', methods=['POST'])
@login_required
def upload_syllabus():
    if 'syllabus_file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    file        = request.files['syllabus_file']
    module_name = request.form.get('module_name', '').strip()
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    original_name = file.filename
    unique_name   = f"syllabus_{int(datetime.utcnow().timestamp())}_{session['user_id']}_{secure_filename(file.filename)}"
    filepath      = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)
    text_content = extract_pdf_text(filepath)
    syl = SyllabusFile(user_id=session['user_id'], filename=unique_name,
                       original_name=original_name, module_name=module_name or None,
                       text_content=text_content)
    db.session.add(syl)
    db.session.commit()
    return jsonify({'id': syl.id, 'original_name': syl.original_name,
                    'module_name': syl.module_name or '', 'has_text': bool(text_content)})


@app.route('/syllabus/list')
@login_required
def list_syllabus():
    files = SyllabusFile.query.filter_by(user_id=session['user_id'])\
                              .order_by(SyllabusFile.upload_date.desc()).all()
    return jsonify([{'id': f.id, 'original_name': f.original_name,
                     'module_name': f.module_name or ''} for f in files])


@app.route('/syllabus/<int:syl_id>/delete', methods=['POST'])
@login_required
def delete_syllabus(syl_id):
    syl = SyllabusFile.query.filter_by(id=syl_id, user_id=session['user_id']).first_or_404()
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], syl.filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    db.session.delete(syl)
    db.session.commit()
    return jsonify({'success': True})


@app.route('/study-plans/ai-generate', methods=['POST'])
@login_required
def ai_generate_schedule():
    import json
    data             = request.get_json()
    modules          = data.get('modules', [])
    start_date       = data.get('start_date', '')
    daily_hours      = data.get('daily_hours', 6)
    sessions_per_day = int(data.get('sessions_per_day', 1))
    hours_per_session = round(daily_hours / sessions_per_day, 1)
    if not modules or not start_date:
        return jsonify({'error': 'Missing modules or start date'}), 400
    if not GEMINI_SUPPORT:
        return jsonify({'error': 'AI not configured. Check GROQ_API_KEY in .env'}), 400
    modules_text = "".join([f"- {m['name']} | Difficulty: {m['difficulty']}/5 | Exam: {m['exam_date']}\n" for m in modules])
    prompt = f"""You are an expert study planner.
Start: {start_date} | {daily_hours}h/day | {sessions_per_day} subject(s)/day | {hours_per_session}h/session
Modules:
{modules_text}
Create schedule up to 20 days. EXACTLY {sessions_per_day} entry/entries per day. Never same module twice per day. 2 topics + tip per session.
RESPOND WITH ONLY a JSON array, no markdown:
[{{"date":"YYYY-MM-DD","module":"Name","hours":{hours_per_session},"topics":["T1","T2"],"tip":"tip"}}]"""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}],
            max_tokens=6000, temperature=0.4)
        raw = response.choices[0].message.content.strip()
        if '```' in raw:
            for part in raw.split('```'):
                p = part.strip()
                if p.startswith('json'): p = p[4:].strip()
                if p.startswith('[') or p.startswith('{'):
                    raw = p; break
        return jsonify({'schedule': json.loads(raw.strip())})
    except json.JSONDecodeError:
        return jsonify({'error': 'AI returned invalid format. Try again.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/study-plans/ai-generate/syllabus', methods=['POST'])
@login_required
def ai_generate_syllabus_schedule():
    import json
    data             = request.get_json()
    pdf_ids          = data.get('pdf_ids') or ([data.get('pdf_id')] if data.get('pdf_id') else [])
    start_date       = data.get('start_date', '')
    daily_hours      = float(data.get('daily_hours', 4))
    sessions_per_day = int(data.get('sessions_per_day', 2))
    exam_dates       = data.get('exam_dates', {})

    if not pdf_ids or not start_date:
        return jsonify({'error': 'Please select at least one syllabus PDF and a start date.'}), 400
    if not GEMINI_SUPPORT:
        return jsonify({'error': 'AI not configured. Check GROQ_API_KEY in .env'}), 400

    syls = SyllabusFile.query.filter(
        SyllabusFile.id.in_(pdf_ids),
        SyllabusFile.user_id == session['user_id']
    ).all()

    if not syls:
        return jsonify({'error': 'Syllabus files not found or no text could be extracted.'}), 404

    combined_text = ""
    for syl in syls:
        label = syl.module_name or syl.original_name
        combined_text += f"\n\n=== SUBJECT: {label} ===\n{(syl.text_content or '')[:4000]}"

    syllabus_text     = combined_text[:10000]
    hours_per_session = round(daily_hours / sessions_per_day, 1)
    subject_names     = ", ".join([syl.module_name or syl.original_name for syl in syls])
    exam_text         = "".join([f"- {m}: exam on {e}\n" for m, e in exam_dates.items()])

    prompt = f"""You are an expert academic study planner. Syllabuses for {len(syls)} subject(s): {subject_names}.

SYLLABUS CONTENT:
{syllabus_text}

STUDENT INFO:
- Start Date: {start_date}
- {daily_hours}h/day | {sessions_per_day} subjects/day | {hours_per_session}h/session
- Exam dates: {exam_text if exam_text else "Not specified — spread over 14 days"}

RULES:
- MAXIMUM 14 days, {sessions_per_day * 14} total JSON entries
- Each day has EXACTLY {sessions_per_day} entries — one per subject, never same subject twice per day
- 1-2 specific topics per session, one short tip
- No markdown, no code fences in response

RESPOND WITH ONLY valid JSON:
{{
  "modules": [{{"name": "Subject", "topics": ["t1", "t2"]}}],
  "schedule": [{{"date": "YYYY-MM-DD", "module": "Subject", "hours": {hours_per_session}, "topics": ["Topic"], "tip": "Tip"}}]
}}"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=6000, temperature=0.3)
        raw = response.choices[0].message.content.strip()

        if '```' in raw:
            for part in raw.split('```'):
                p = part.strip()
                if p.startswith('json'): p = p[4:].strip()
                if p.startswith('{'):
                    raw = p; break
        raw = raw.strip()

        if not raw.endswith('}'):
            last_entry = raw.rfind('},')
            if last_entry == -1: last_entry = raw.rfind('}')
            if last_entry > 0:
                raw = raw[:last_entry + 1] + '\n  ]\n}'

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            sched_start = raw.find('"schedule"')
            if sched_start > 0:
                arr_start = raw.find('[', sched_start)
                last_obj  = raw.rfind('}')
                if arr_start > 0 and last_obj > arr_start:
                    result = {'modules': [], 'schedule': json.loads(raw[arr_start:last_obj + 1] + ']')}
                else:
                    raise
            else:
                raise

        return jsonify(result)

    except json.JSONDecodeError:
        return jsonify({'error': 'AI returned invalid format. Try again. If this repeats, reduce subjects per day.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/study-plans/ai-generate/check-conflicts', methods=['POST'])
@login_required
def check_conflicts():
    """Return conflict info only for the exact subjects being saved on those dates."""
    data           = request.get_json()
    dates          = data.get('dates', [])
    day_hours      = float(data.get('day_hours', 6))
    new_subjects   = data.get('new_subjects', [])   # subjects in the incoming schedule
    uid            = session['user_id']

    conflicts         = []   # dates where the SAME subject already exists
    existing_subjects = []   # which of the new subjects are already saved on those dates
    hours_used        = {}   # date_str -> hours already saved
    fully_booked      = []   # dates with no hours left at all

    for d in dates:
        try:
            date_obj = datetime.strptime(d, '%Y-%m-%d').date()
            all_plans = StudyPlan.query.filter_by(user_id=uid, schedule_date=date_obj).all()
            if not all_plans:
                continue

            used = sum(p.study_hours for p in all_plans)
            hours_used[d] = used

            # Only flag a conflict if the SAME subject is already on this date
            clashing = [p.subject for p in all_plans
                        if not new_subjects or p.subject in new_subjects]
            if clashing:
                conflicts.append(date_obj.strftime('%b %d'))
                for subj in clashing:
                    if subj not in existing_subjects:
                        existing_subjects.append(subj)

            if used >= day_hours:
                fully_booked.append(date_obj.strftime('%b %d'))
        except Exception:
            pass

    return jsonify({
        'conflicts':         conflicts,
        'existing_subjects': existing_subjects,
        'hours_used':        hours_used,
        'fully_booked':      fully_booked,
        'day_hours':         day_hours
    })


@app.route('/study-plans/ai-generate/save', methods=['POST'])
@login_required
def save_ai_schedule():
    data     = request.get_json()
    schedule = data.get('schedule', [])
    mode     = data.get('mode', 'replace')   # 'merge' or 'replace'
    uid      = session['user_id']
    saved    = 0

    if not schedule:
        return jsonify({'saved': 0, 'skipped': 0, 'success': True, 'received': 0})

    # Parse all incoming dates
    incoming_dates = []
    for day in schedule:
        try:
            incoming_dates.append(datetime.strptime(day['date'], '%Y-%m-%d').date())
        except Exception:
            pass

    if mode == 'replace' and incoming_dates:
        # Wipe existing sessions for these dates before saving new ones
        StudyPlan.query.filter(
            StudyPlan.user_id == uid,
            StudyPlan.schedule_date.in_(incoming_dates)
        ).delete(synchronize_session=False)
        db.session.flush()

    # For merge mode: precompute hours already used per date
    hours_used_map = {}
    if mode == 'merge':
        for d in incoming_dates:
            used = db.session.query(
                db.func.sum(StudyPlan.study_hours)
            ).filter_by(user_id=uid, schedule_date=d).scalar() or 0.0
            hours_used_map[d] = used

    day_hours = float(data.get('day_hours', 6))  # full day hours from settings

    for day in schedule:
        try:
            date_obj = datetime.strptime(day['date'], '%Y-%m-%d').date()
            subject  = str(day.get('module') or day.get('subject', '')).strip()
            hours    = float(day.get('hours') or day.get('dayTotal') or 2)
            if not subject:
                continue

            if mode == 'merge':
                # Skip if exact same subject already on that date
                exists = StudyPlan.query.filter_by(
                    user_id=uid, schedule_date=date_obj, subject=subject
                ).first()
                if exists:
                    continue
                # Scale hours to fit remaining time in the day
                used      = hours_used_map.get(date_obj, 0.0)
                remaining = day_hours - used
                if remaining <= 0:
                    continue   # day is fully booked, skip
                # Cap this session's hours to what's left
                hours = min(hours, round(remaining, 1))
                # Track for subsequent sessions on same date within this batch
                hours_used_map[date_obj] = used + hours

            plan = StudyPlan(
                user_id       = uid,
                subject       = subject,
                study_hours   = hours,
                schedule_date = date_obj
            )
            db.session.add(plan)
            saved += 1
        except Exception:
            continue

    db.session.commit()
    return jsonify({'saved': saved, 'skipped': 0, 'success': True, 'received': len(schedule)})


@app.route('/study-plans/finish-session', methods=['POST'])
@login_required
def finish_session():
    """Mark a saved session as completed by matching date + subject name."""
    data    = request.get_json()
    date_str = data.get('date', '')
    subject  = data.get('subject', '').strip()
    undo     = data.get('undo', False)
    if not date_str or not subject:
        return jsonify({'success': False, 'error': 'Missing date or subject'}), 400
    try:
        sched_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid date'}), 400
    plan = StudyPlan.query.filter_by(
        user_id=session['user_id'],
        schedule_date=sched_date,
        subject=subject
    ).first()
    if not plan:
        return jsonify({'success': False, 'error': 'Session not found in saved plans'})
    plan.completed = not undo
    plan.skipped   = False
    db.session.commit()
    return jsonify({'success': True, 'completed': plan.completed})


@app.route('/pdf-notes')
@login_required
def pdf_notes():
    pdfs = PDFNote.query.filter_by(user_id=session['user_id'])\
                        .order_by(PDFNote.upload_date.desc()).all()
    return render_template('pdf_notes.html', pdfs=pdfs)


@app.route('/pdf-notes/upload', methods=['POST'])
@login_required
def upload_pdf():
    if 'pdf_file' not in request.files:
        flash('No file selected.', 'danger')
        return redirect(url_for('pdf_notes'))
    file    = request.files['pdf_file']
    subject = request.form.get('subject', '').strip()
    if file.filename == '':
        flash('No file selected.', 'danger')
        return redirect(url_for('pdf_notes'))
    if not allowed_file(file.filename):
        flash('Only PDF files are allowed.', 'danger')
        return redirect(url_for('pdf_notes'))
    original_name = file.filename
    unique_name   = f"{int(datetime.utcnow().timestamp())}_{session['user_id']}_{secure_filename(file.filename)}"
    filepath      = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)
    text_content = extract_pdf_text(filepath)
    pdf = PDFNote(user_id=session['user_id'], filename=unique_name, original_name=original_name,
                  subject=subject or None, text_content=text_content)
    db.session.add(pdf)
    db.session.commit()
    if text_content:
        flash(f'"{original_name}" uploaded and ready!', 'success')
    else:
        flash(f'"{original_name}" uploaded but text extraction failed. Try a digital PDF.', 'warning')
    return redirect(url_for('pdf_notes'))


@app.route('/pdf-notes/<int:pdf_id>/delete', methods=['POST'])
@login_required
def delete_pdf(pdf_id):
    pdf = PDFNote.query.filter_by(id=pdf_id, user_id=session['user_id']).first_or_404()
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], pdf.filename)
    if os.path.exists(filepath): os.remove(filepath)
    db.session.delete(pdf)
    db.session.commit()
    flash('PDF deleted.', 'info')
    return redirect(url_for('pdf_notes'))


@app.route('/chatbot')
@login_required
def chatbot():
    uid          = session['user_id']
    pdfs         = PDFNote.query.filter_by(user_id=uid).order_by(PDFNote.upload_date.desc()).all()
    selected_pdf = None
    pdf_id       = request.args.get('pdf_id', type=int)
    if pdf_id:
        selected_pdf = PDFNote.query.filter_by(id=pdf_id, user_id=uid).first()
    chat_key     = f'chat_{pdf_id}' if pdf_id else 'chat_general'
    chat_history = session.get(chat_key, [])
    return render_template('chatbot.html', pdfs=pdfs, selected_pdf=selected_pdf,
                           chat_history=chat_history, gemini_enabled=GEMINI_SUPPORT)


@app.route('/chatbot/query', methods=['POST'])
@login_required
def chatbot_query():
    data   = request.get_json()
    query  = (data.get('query') or '').strip()
    pdf_id = data.get('pdf_id')
    if not query:
        return jsonify({'answer': 'Please type a question.', 'error': True})
    pdf_text = ''
    if pdf_id:
        pdf = PDFNote.query.filter_by(id=pdf_id, user_id=session['user_id']).first()
        if pdf and pdf.text_content:
            pdf_text = pdf.text_content
    chat_key     = f'chat_{pdf_id}' if pdf_id else 'chat_general'
    chat_history = session.get(chat_key, [])
    answer = ask_gemini(pdf_text, query, chat_history)
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})
    if len(chat_history) > 20:
        chat_history = chat_history[-20:]
    session[chat_key] = chat_history
    session.modified  = True
    return jsonify({'answer': answer, 'error': False})


@app.route('/chatbot/clear', methods=['POST'])
@login_required
def clear_chat():
    data   = request.get_json()
    pdf_id = data.get('pdf_id')
    key    = f'chat_{pdf_id}' if pdf_id else 'chat_general'
    session.pop(key, None)
    session.modified = True
    return jsonify({'success': True})


@app.route('/admin')
@admin_required
def admin_panel():
    users       = User.query.all()
    total_plans = StudyPlan.query.count()
    total_pdfs  = PDFNote.query.count()
    return render_template('admin.html', users=users,
                           total_plans=total_plans, total_pdfs=total_pdfs)


@app.route('/admin/delete-user/<int:user_id>', methods=['POST'])
@admin_required
def admin_delete_user(user_id):
    if user_id == session['user_id']:
        flash("You can't delete your own account.", 'danger')
        return redirect(url_for('admin_panel'))
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    flash(f'User "{user.name}" deleted.', 'info')
    return redirect(url_for('admin_panel'))


@app.route('/study-plans/extract-syllabus', methods=['POST'])
@login_required
def extract_syllabus_topics():
    import json
    if not GEMINI_SUPPORT:
        return jsonify({'error': 'AI not configured. Check GROQ_API_KEY in .env'}), 400

    if 'syllabus_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['syllabus_file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Please upload a valid PDF file'}), 400

    tmp_name = f"tmp_syl_{int(datetime.utcnow().timestamp())}_{session['user_id']}.pdf"
    tmp_path = os.path.join(app.config['UPLOAD_FOLDER'], tmp_name)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(tmp_path)

    try:
        text = extract_pdf_text(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not text.strip():
        return jsonify({'error': 'Could not extract text from this PDF. Try a digital (non-scanned) PDF.'}), 400

    syllabus_text = text[:12000]

    prompt = f"""You are a syllabus analyser. Read this syllabus and extract topics MODULE BY MODULE.

SYLLABUS TEXT:
{syllabus_text}

Return ONLY a valid JSON object — no markdown, no explanation, no text before or after:
{{
  "subjects": [
    {{
      "subject": "Module 1: Introduction to compilers and lexical analysis",
      "priority": "high",
      "topics": [
        "Analysis of the source program",
        "Analysis and synthesis phases",
        "Phases of a compiler"
      ]
    }},
    {{
      "subject": "Module 2: Introduction to Syntax Analysis",
      "priority": "high",
      "topics": [
        "Role of the Syntax Analyser",
        "Syntax error handling"
      ]
    }}
  ]
}}

Rules:
- Create ONE subject entry PER MODULE/UNIT found in the syllabus. Do NOT merge all modules into one.
- Use the exact module/unit name as the subject (e.g. "Module 1: ...", "Unit 2: ...").
- List only the topics belonging to that specific module under its topics array.
- Use EXACT topic names from the syllabus — do not paraphrase.
- Include ALL topics from every module, no skipping.
- priority = "high" for all modules unless clearly marked as optional/minor."""

    # Try primary model; fall back to smaller models on rate-limit (429)
    use_fallback = request.form.get('fallback') == '1'
    MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"]
    model_list = MODELS[1:] if use_fallback else MODELS

    for model in model_list:
        try:
            response = groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a syllabus analyser. Respond ONLY with valid JSON. No markdown, no explanation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.2
            )
            raw = response.choices[0].message.content.strip()
            if '```' in raw:
                for part in raw.split('```'):
                    p = part.strip()
                    if p.startswith('json'): p = p[4:].strip()
                    if p.startswith('{'): raw = p; break
            raw = raw.strip()
            data = json.loads(raw)
            subjects = data.get('subjects', [])
            if not subjects:
                return jsonify({'error': 'No subjects found in the PDF. Try a different syllabus file.'}), 400
            return jsonify({'subjects': subjects})
        except json.JSONDecodeError:
            return jsonify({'error': 'AI returned invalid format. Try again.'}), 500
        except Exception as e:
            if '429' in str(e) or 'rate_limit' in str(e):
                continue   # try next model
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'rate_limit: Groq daily token limit reached on all models. Please wait and try again later.'}), 429


@app.route('/study-plans/generate-timetable', methods=['POST'])
@login_required
def generate_full_timetable():
    import json

    if not GEMINI_SUPPORT:
        return jsonify({'error': 'AI not configured. Check GROQ_API_KEY in .env'}), 400

    data       = request.get_json()
    subjects   = data.get('subjects', [])
    start_date = data.get('start_date', '')
    num_days   = int(data.get('num_days', 7))
    hours      = int(data.get('hours_per_day', 6))
    start_time = data.get('start_time', '08:00')
    break_days = int(data.get('break_days', 1))
    extra      = data.get('extra_instructions', '').strip()
    num_subjects = int(data.get('num_subjects', len(subjects)))

    if not subjects:
        return jsonify({'error': 'No subjects provided'}), 400
    if not start_date:
        return jsonify({'error': 'Start date is required'}), 400
    if num_days < 1 or num_days > 30:
        return jsonify({'error': 'Number of days must be between 1 and 30'}), 400

    try:
        s_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'error': 'Invalid date format'}), 400

    e_dt = s_dt + timedelta(days=num_days - 1)

    rest_day_names = {1: ['Sunday'], 2: ['Saturday', 'Sunday'], 0: []}.get(break_days, ['Sunday'])

    # Auto-detect study mode
    # 1 subject uploaded → its "subjects" are actually modules → cover one module per day (deep focus per module)
    # 2+ subjects uploaded → rotate between subjects each day
    if num_subjects == 1:
        # Sort modules by name so Module 1 < Module 2 < Module 3 etc.
        import re as _re
        def _mod_sort_key(s):
            m = _re.search(r'(\d+)', s.get('subject', ''))
            return int(m.group(1)) if m else 9999
        subjects = sorted(subjects, key=_mod_sort_key)

        study_mode_instruction = (
            f"There is 1 subject split into {len(subjects)} modules. "
            "STRICT RULE: Schedule modules in the EXACT order listed below (Module 1 first, then Module 2, etc.). "
            "NEVER schedule a later module before an earlier one. "
            "Complete ALL topics of Module 1 before any session from Module 2 appears. "
            "Do NOT mix modules on the same day — each day covers only one module."
        )
    else:
        study_mode_instruction = (
            f"There are {num_subjects} subjects. "
            "STRICT RULE: Within EACH study day, alternate sessions between subjects. "
            "For example with 2 subjects A and B and 4 sessions: A → B → A → B. "
            "With 3 subjects A, B, C and 6 sessions: A → B → C → A → B → C. "
            "Every subject must appear at least once per study day. "
            f"Spread topics of each subject evenly across all {num_days} study days."
        )

    # Build subject + topics block for the prompt (ordered)
    subj_lines = []
    for i, s in enumerate(subjects):
        topics = s.get('topics', [])[:15]
        subj_lines.append(f"{i+1}. {s['subject']} [{s.get('priority','medium')} priority]  ← schedule this AFTER all previous modules")
        if topics:
            subj_lines.append("   Topics (in order): " + " | ".join(topics))
    subj_block    = "\n".join(subj_lines)
    subject_names = ", ".join(s['subject'] for s in subjects)

    def parse_json(raw):
        raw = raw.strip()
        # Strip markdown fences
        if "```" in raw:
            for part in raw.split("```"):
                p = part.strip().lstrip("json").strip()
                if p.startswith("["):
                    raw = p; break
        raw = raw.strip()
        # Try parsing as-is first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # Try to repair truncated JSON — find last complete object
        last_obj = raw.rfind("}]")
        if last_obj > 0:
            try:
                return json.loads(raw[:last_obj + 2])
            except json.JSONDecodeError:
                pass
        last_obj = raw.rfind("}")
        if last_obj > 0:
            try:
                return json.loads(raw[:last_obj + 1] + "]")
            except json.JSONDecodeError:
                pass
        raise json.JSONDecodeError("Could not repair JSON", raw, 0)

    # Keep topics list short to avoid token overflow
    subj_block_short = []
    for i, s in enumerate(subjects):
        topics = s.get('topics', [])[:8]   # max 8 topics per module in prompt
        subj_block_short.append(f"{i+1}. {s['subject']} [{s.get('priority','medium')}]")
        if topics:
            subj_block_short.append("   Topics: " + " | ".join(topics))
    subj_block = "\n".join(subj_block_short)

    # Generate all days in one single prompt
    prompt = f"""You are a study planner. Generate a study timetable for EXACTLY {num_days} days.

START DATE: {s_dt}
END DATE: {e_dt}
HOURS PER DAY: {hours}h starting at {start_time}
REST DAYS: {", ".join(rest_day_names) if rest_day_names else "none"}

STUDY MODE: {study_mode_instruction}

SUBJECTS AND TOPICS:
{subj_block}

RULES:
- EXACTLY {num_days} entries, one per date from {s_dt} to {e_dt}.
- Rest days: type="rest", sessions=[].
- Study days: type="study", sessions covering subjects.
- Keep topics array SHORT (max 2 per session) to stay within token limits.
- Each session: time slot, subject, durationHours, topics (max 2), note (max 8 words).
{("- Extra: " + extra) if extra else ""}

OUTPUT ONLY valid compact JSON array, no markdown:
[{{"date":"YYYY-MM-DD","dayName":"Monday","type":"study","totalHours":{hours},"sessions":[{{"time":"09:00-11:00","subject":"Name","durationHours":2,"topics":["T1","T2"],"note":"tip"}}]}}]"""

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a study planner. Output ONLY valid compact JSON. No markdown, no explanation, no extra text."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=8000,
            temperature=0.3
        )
        raw      = resp.choices[0].message.content.strip()
        all_days = parse_json(raw)

        # Safety: trim to num_days if AI returned more
        all_days = all_days[:num_days]

        study_days  = [d for d in all_days if d.get('type') == 'study']
        total_hours = sum(d.get('totalHours', 0) for d in study_days)

        return jsonify({
            "title":          f"Study Plan — {subject_names}",
            "startDate":      start_date,
            "examDate":       e_dt.strftime('%Y-%m-%d'),
            "totalStudyDays": len(study_days),
            "totalHours":     total_hours,
            "numSubjects":    num_subjects,
            "subjects":       [s['subject'] for s in subjects],
            "days":           all_days
        })

    except json.JSONDecodeError:
        return jsonify({'error': 'AI returned invalid format. Try again.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/my-timetable')
@login_required
def my_timetable():
    return render_template('my_timetable.html')


@app.route('/study-plans/save-timetable', methods=['POST'])
@login_required
def save_timetable():
    import json
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    uid   = session['user_id']
    title = data.get('title', 'My Timetable')
    # Always create a new timetable entry (support multiple)
    new_tt = SavedTimetable(user_id=uid, title=title, timetable_json=json.dumps(data))
    db.session.add(new_tt)
    db.session.commit()
    return jsonify({'success': True, 'id': new_tt.id})


@app.route('/study-plans/load-timetable', methods=['GET'])
@login_required
def load_timetable():
    import json
    uid  = session['user_id']
    rows = SavedTimetable.query.filter_by(user_id=uid).order_by(SavedTimetable.created_at.desc()).all()
    if not rows:
        return jsonify({'timetables': []})
    timetables = []
    for row in rows:
        data = json.loads(row.timetable_json)
        timetables.append({
            'id':         row.id,
            'title':      row.title,
            'created_at': row.created_at.strftime('%b %d, %Y'),
            'timetable':  data
        })
    return jsonify({'timetables': timetables})


@app.route('/study-plans/delete-timetable', methods=['POST'])
@login_required
def delete_timetable():
    import json
    uid  = session['user_id']
    data = request.get_json(silent=True) or {}
    tid  = data.get('id')
    if tid:
        row = SavedTimetable.query.filter_by(id=tid, user_id=uid).first()
    else:
        row = SavedTimetable.query.filter_by(user_id=uid).order_by(SavedTimetable.created_at.desc()).first()
    if row:
        db.session.delete(row)
        db.session.commit()
    return jsonify({'success': True})


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # ── Migrate: add 'skipped' column if it doesn't exist yet ──
        from sqlalchemy import text
        with db.engine.connect() as conn:
            try:
                conn.execute(text("ALTER TABLE study_plans ADD COLUMN skipped BOOLEAN DEFAULT 0"))
                conn.commit()
                print("Migration: 'skipped' column added.")
            except Exception:
                pass  # Column already exists — that's fine
            try:
                conn.execute(text("""CREATE TABLE IF NOT EXISTS saved_timetables (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id),
                    title VARCHAR(255) NOT NULL,
                    timetable_json TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)"""))
                conn.commit()
                print("Migration: 'saved_timetables' table ready.")
            except Exception:
                pass
        print("Database ready.")
        print(f"Groq Key: {'YES' if os.environ.get('GROQ_API_KEY') else 'NO - check .env'}")
        print(f"Groq AI: {'enabled' if GEMINI_SUPPORT else 'disabled'}")
        print(f"OCR: {'enabled' if OCR_SUPPORT else 'disabled'}")
        print("Starting Smart Study Planner...")
    app.run(debug=True)
