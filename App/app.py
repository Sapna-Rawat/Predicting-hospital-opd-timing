from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text, event
from sqlalchemy.exc import IntegrityError
from werkzeug.security import generate_password_hash, check_password_hash
from pathlib import Path
from datetime import datetime, date, time, timedelta
import functools
import os
import secrets

# Suppress TensorFlow verbose logging when lazily imported
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ============================================================
# 1. INITIALIZE FLASK APPLICATION
# ============================================================

app = Flask(__name__)

project_folder = Path(__file__).resolve().parent.parent
database_path = Path(os.environ.get("OPD_DATABASE_PATH", str(project_folder / "database" / "opd_management.db")))
models_folder = project_folder / "Models"

# Use environment variable for secret key; generate a stable local instance key if unset
_secret_key = os.environ.get("FLASK_SECRET_KEY")
if not _secret_key:
    _instance_key_file = Path(__file__).resolve().parent / ".flask_secret_key"
    if _instance_key_file.exists():
        _secret_key = _instance_key_file.read_text(encoding="utf-8").strip()
    else:
        _secret_key = secrets.token_hex(32)
        try:
            _instance_key_file.write_text(_secret_key, encoding="utf-8")
        except Exception:
            pass
app.config["SECRET_KEY"] = _secret_key


# ============================================================
# 2. DATABASE CONFIGURATION
# ============================================================

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + database_path.as_posix()
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# ============================================================
# 3. ENABLE SQLITE FOREIGN KEYS
# ============================================================

with app.app_context():
    engine = db.engine

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, record):
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


# ============================================================
# 4. DATABASE MODELS (Exact schemas matching existing SQLite)
# ============================================================

class Patient(db.Model):
    __tablename__ = "patients"

    patient_id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    phone = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, server_default=db.func.current_timestamp())


class Department(db.Model):
    __tablename__ = "departments"

    department_id = db.Column(db.Integer, primary_key=True)
    department_name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)


class Doctor(db.Model):
    __tablename__ = "doctors"

    doctor_id = db.Column(db.Integer, primary_key=True)
    doctor_name = db.Column(db.String(100), nullable=False)
    department_id = db.Column(db.Integer, nullable=False)
    specialization = db.Column(db.String(100))
    availability_status = db.Column(db.String(30), default="Available")


class Appointment(db.Model):
    __tablename__ = "appointments"

    appointment_id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, nullable=False)
    department_id = db.Column(db.Integer, nullable=False)
    doctor_id = db.Column(db.Integer)
    appointment_date = db.Column(db.Date, nullable=False)
    appointment_time = db.Column(db.Time)
    registration_time = db.Column(db.DateTime, server_default=db.func.current_timestamp())
    appointment_status = db.Column(db.String(30), default="Booked")


class QueueEntry(db.Model):
    __tablename__ = "queue"

    queue_id = db.Column(db.Integer, primary_key=True)
    appointment_id = db.Column(db.Integer, nullable=False)
    token_number = db.Column(db.Integer, nullable=False)
    queue_status = db.Column(db.String(30), default="Booked")
    arrival_time = db.Column(db.DateTime)
    consultation_start = db.Column(db.DateTime)
    consultation_end = db.Column(db.DateTime)


class PredictionRecord(db.Model):
    __tablename__ = "predictions"

    prediction_id = db.Column(db.Integer, primary_key=True)
    appointment_id = db.Column(db.Integer, nullable=False)
    predicted_wait_minutes = db.Column(db.Float)
    predicted_service_minutes = db.Column(db.Float)
    predicted_consultation_time = db.Column(db.DateTime)
    predicted_at = db.Column(db.DateTime, server_default=db.func.current_timestamp())


class HospitalStaff(db.Model):
    __tablename__ = "hospital_staff"

    staff_id = db.Column(db.Integer, primary_key=True)
    staff_name = db.Column(db.Text, nullable=False)
    email = db.Column(db.Text, unique=True, nullable=False)
    password_hash = db.Column(db.Text, nullable=False)
    role = db.Column(db.Text, default="Staff")


# ============================================================
# 5. DEEP LEARNING MODEL LAZY LOADING & INFERENCE MODULE
# ============================================================

_WAIT_MODEL = None
_WAIT_FEAT_SCALER = None
_WAIT_TARGET_SCALER = None
_WAIT_FEATURES = None

_SERVICE_MODEL = None
_SERVICE_TARGET_SCALER = None
_SERVICE_FEATURES = None

# Department baseline service times (minutes) used only when LSTM inputs are unavailable
DEPT_SERVICE_BASELINES = {
    1: 8.5,   # General Medicine
    2: 12.0,  # Cardiology
    3: 9.0,   # Dermatology
    4: 10.5,  # Orthopedics
    5: 11.0,  # Pediatrics
    6: 12.5   # Gynecology
}

ALLOWED_STATUS_TRANSITIONS = {
    "Booked": {"Booked", "Reached", "Waiting", "In Consultation", "Cancelled"},
    "Reached": {"Reached", "Waiting", "In Consultation", "Completed", "Cancelled"},
    "Waiting": {"Waiting", "In Consultation", "Completed", "Cancelled"},
    "In Consultation": {"In Consultation", "Completed", "Cancelled"},
    "Completed": {"Completed"},
    "Cancelled": {"Cancelled"},
}


def load_wait_model_assets():
    global _WAIT_MODEL, _WAIT_FEAT_SCALER, _WAIT_TARGET_SCALER, _WAIT_FEATURES
    if _WAIT_MODEL is None:
        try:
            import joblib
            from tensorflow.keras.models import load_model
            m_path = models_folder / "opd_wait_lstm.keras"
            fs_path = models_folder / "feature_scaler.pkl"
            ts_path = models_folder / "target_scaler.pkl"
            feat_path = models_folder / "selected_features.pkl"
            if m_path.exists() and fs_path.exists() and ts_path.exists() and feat_path.exists():
                _WAIT_MODEL = load_model(m_path)
                _WAIT_FEAT_SCALER = joblib.load(fs_path)
                _WAIT_TARGET_SCALER = joblib.load(ts_path)
                _WAIT_FEATURES = joblib.load(feat_path)
        except Exception as e:
            print("Failed to load OPD Wait LSTM model:", e)
    return _WAIT_MODEL, _WAIT_FEAT_SCALER, _WAIT_TARGET_SCALER, _WAIT_FEATURES


def load_service_model_assets():
    """
    Loads Model 2 (HanguData consultation duration LSTM) and the CORRECT target scaler
    (hangu_target_scaler.pkl) + feature column order (hangu_feature_columns.pkl).
    Never uses hangu_feature_scaler.pkl.
    """
    global _SERVICE_MODEL, _SERVICE_TARGET_SCALER, _SERVICE_FEATURES
    if _SERVICE_MODEL is None:
        try:
            import joblib
            from tensorflow.keras.models import load_model
            m_path = models_folder / "hangu_service_lstm.keras"
            ts_path = models_folder / "hangu_target_scaler.pkl"
            feat_path = models_folder / "hangu_feature_columns.pkl"
            if m_path.exists() and ts_path.exists() and feat_path.exists():
                _SERVICE_MODEL = load_model(m_path)
                _SERVICE_TARGET_SCALER = joblib.load(ts_path)
                _SERVICE_FEATURES = joblib.load(feat_path)
        except Exception as e:
            print("Failed to load Hangu Service LSTM model:", e)
    return _SERVICE_MODEL, _SERVICE_TARGET_SCALER, _SERVICE_FEATURES


def _compute_waitdata_feature_row(item_q, item_appt, all_records):
    """
    Reproduces the exact 21 WaitData features from real SQLite queue & appointment timestamps:
    AheadCount, LineCount0..4, FlowCount2/4, InProgressSize,
    NumCompletedInLastW1..3, NumCustomersInLastW1..3, AvgWaitLastW1..3,
    Month, DayOfWeek, DayOfYear.
    Returns None if rolling wait windows have zero observed waits (cannot reproduce AvgWaitLastW1..3).
    """
    t_arr = item_q.arrival_time
    if t_arr is None:
        return None

    # AheadCount: scheduled/booked patients before this patient in the same department & date
    ahead_count = 0
    for q_other, a_other in all_records:
        if q_other.queue_id == item_q.queue_id or q_other.queue_status == "Cancelled":
            continue
        if a_other.appointment_time and item_appt.appointment_time:
            if a_other.appointment_time < item_appt.appointment_time or (
                a_other.appointment_time == item_appt.appointment_time
                and q_other.token_number < item_q.token_number
            ):
                ahead_count += 1
        elif q_other.token_number < item_q.token_number:
            ahead_count += 1

    def count_in_line_at(ref_time):
        cnt = 0
        for q_other, _ in all_records:
            if q_other.queue_status == "Cancelled" or q_other.arrival_time is None:
                continue
            if q_other.arrival_time <= ref_time and (
                q_other.consultation_start is None or q_other.consultation_start > ref_time
            ):
                cnt += 1
        return float(cnt)

    line_counts = [
        count_in_line_at(t_arr - timedelta(minutes=15 * k)) for k in range(5)
    ]

    w30 = t_arr - timedelta(minutes=30)
    w60 = t_arr - timedelta(minutes=60)
    w120 = t_arr - timedelta(minutes=120)

    flow_count_2 = sum(
        1 for q_other, _ in all_records
        if q_other.consultation_start is not None and w30 <= q_other.consultation_start <= t_arr
    )
    flow_count_4 = sum(
        1 for q_other, _ in all_records
        if q_other.consultation_start is not None and w60 <= q_other.consultation_start <= t_arr
    )

    in_progress_size = sum(
        1 for q_other, _ in all_records
        if q_other.consultation_start is not None
        and q_other.consultation_start <= t_arr
        and (q_other.consultation_end is None or q_other.consultation_end > t_arr)
    )

    num_completed_w1 = sum(
        1 for q_other, _ in all_records
        if q_other.consultation_end is not None and w30 <= q_other.consultation_end <= t_arr
    )
    num_completed_w2 = sum(
        1 for q_other, _ in all_records
        if q_other.consultation_end is not None and w60 <= q_other.consultation_end <= t_arr
    )
    num_completed_w3 = sum(
        1 for q_other, _ in all_records
        if q_other.consultation_end is not None and w120 <= q_other.consultation_end <= t_arr
    )

    num_cust_w1 = sum(
        1 for q_other, _ in all_records
        if q_other.arrival_time is not None and w30 <= q_other.arrival_time <= t_arr
    )
    num_cust_w2 = sum(
        1 for q_other, _ in all_records
        if q_other.arrival_time is not None and w60 <= q_other.arrival_time <= t_arr
    )
    num_cust_w3 = sum(
        1 for q_other, _ in all_records
        if q_other.arrival_time is not None and w120 <= q_other.arrival_time <= t_arr
    )

    def avg_wait_since(since_time):
        waits = [
            max(0.0, (q_other.consultation_start - q_other.arrival_time).total_seconds() / 60.0)
            for q_other, _ in all_records
            if q_other.arrival_time is not None
            and q_other.consultation_start is not None
            and since_time <= q_other.consultation_start <= t_arr
        ]
        return (sum(waits) / len(waits)) if waits else None

    avg_w3 = avg_wait_since(w120)
    if avg_w3 is None:
        return None
    avg_w2 = avg_wait_since(w60)
    if avg_w2 is None:
        avg_w2 = avg_w3
    avg_w1 = avg_wait_since(w30)
    if avg_w1 is None:
        avg_w1 = avg_w2

    return {
        "AheadCount": float(ahead_count),
        "LineCount0": line_counts[0],
        "LineCount1": line_counts[1],
        "LineCount2": line_counts[2],
        "LineCount3": line_counts[3],
        "LineCount4": line_counts[4],
        "FlowCount2": float(flow_count_2),
        "FlowCount4": float(flow_count_4),
        "InProgressSize": float(in_progress_size),
        "NumCompletedInLastW1": float(num_completed_w1),
        "NumCompletedInLastW2": float(num_completed_w2),
        "NumCompletedInLastW3": float(num_completed_w3),
        "NumCustomersInLastW1": float(num_cust_w1),
        "NumCustomersInLastW2": float(num_cust_w2),
        "NumCustomersInLastW3": float(num_cust_w3),
        "AvgWaitLastW1": avg_w1,
        "AvgWaitLastW2": avg_w2,
        "AvgWaitLastW3": avg_w3,
        "Month": float(t_arr.month),
        "DayOfWeek": float(t_arr.weekday()),
        "DayOfYear": float(t_arr.timetuple().tm_yday),
    }


def predict_patient_wait_time(department_id, appointment_date, patients_ahead):
    """
    Model 1: Wait-Time Prediction (WaitData LSTM, shape (1, 5, 21)).
    Runs live LSTM ONLY when >= 5 real sequential arrival records with valid
    timestamps exist for this department and date so all 21 features can be computed
    without fabricated constants. Otherwise returns a clearly labelled queue-based estimate.
    Returns: (wait_minutes, is_lstm, method_label)
    """
    day_records = (
        db.session.query(QueueEntry, Appointment)
        .join(Appointment, QueueEntry.appointment_id == Appointment.appointment_id)
        .filter(
            Appointment.department_id == department_id,
            Appointment.appointment_date == appointment_date,
        )
        .order_by(QueueEntry.arrival_time.asc(), QueueEntry.token_number.asc())
        .all()
    )

    arrived_records = [
        (q, a) for q, a in day_records
        if q.arrival_time is not None and q.queue_status != "Cancelled"
    ]

    if len(arrived_records) >= 5:
        model, feat_scaler, target_scaler, selected_features = load_wait_model_assets()
        if model is not None and selected_features is not None:
            try:
                import numpy as np
                import pandas as pd
                last_five = arrived_records[-5:]
                seq = []
                valid_seq = True
                for q_item, a_item in last_five:
                    row_dict = _compute_waitdata_feature_row(q_item, a_item, day_records)
                    if row_dict is None:
                        valid_seq = False
                        break
                    seq.append([row_dict[col] for col in selected_features])

                if valid_seq and len(seq) == 5:
                    seq_df = pd.DataFrame(seq, columns=selected_features)
                    seq_scaled = feat_scaler.transform(seq_df)
                    seq_input = seq_scaled.reshape(1, 5, len(selected_features))

                    pred_scaled = model.predict(seq_input, verbose=0)
                    pred_unscaled = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))
                    pred_minutes = float(pred_unscaled[0][0])
                    if np.isfinite(pred_minutes):
                        return max(1.0, round(pred_minutes, 1)), True, "WaitData LSTM (Live 5x21 Sequence)"
            except Exception as ex:
                print("LSTM Wait prediction inference exception:", ex)

    # Clearly labelled queue-based baseline estimate when live 5-arrival sequence is unavailable
    avg_service = DEPT_SERVICE_BASELINES.get(department_id, 10.0)
    in_progress = sum(1 for q, _ in day_records if q.queue_status == "In Consultation")
    baseline_wait = (patients_ahead * avg_service) + (avg_service * 0.5 if in_progress > 0 else 0.0)
    if patients_ahead == 0 and in_progress == 0:
        baseline_wait = 5.0
    return round(baseline_wait, 1), False, "Queue-Based Estimate (Insufficient 5-arrival live sequence)"


def predict_patient_service_duration(department_id, appointment_date=None, appointment_time=None, gender=None, address=None):
    """
    Model 2: Consultation / Service Duration Prediction (HanguData LSTM).
    Inputs:
      - history_input: (1, 5, 1) of the 5 previous completed durations, SCALED via hangu_target_scaler.pkl
      - current_input: (1, 28) matching hangu_feature_columns.pkl exact order
    Runs live LSTM ONLY when >= 5 completed consultations with start/end timestamps exist
    AND valid patient demographic inputs (gender in {'M','F'}, address in allowed categories) are provided.
    Otherwise returns a clearly labelled historical reference estimate.
    Returns: (service_minutes, is_lstm, method_label)
    """
    completed_query = (
        db.session.query(QueueEntry)
        .join(Appointment, QueueEntry.appointment_id == Appointment.appointment_id)
        .filter(
            Appointment.department_id == department_id,
            QueueEntry.queue_status == "Completed",
            QueueEntry.consultation_start.isnot(None),
            QueueEntry.consultation_end.isnot(None),
        )
        .order_by(QueueEntry.consultation_end.asc(), QueueEntry.queue_id.asc())
        .all()
    )

    valid_genders = {"M", "F"}
    valid_addresses = {"In the city", "Out of city", "Out of province", "Unknown"}

    if len(completed_query) >= 5 and gender in valid_genders and address in valid_addresses:
        model, target_scaler, feature_cols = load_service_model_assets()
        if model is not None and target_scaler is not None and feature_cols is not None:
            try:
                import numpy as np
                import pandas as pd
                all_durations = [
                    max(1.0, (e.consultation_end - e.consultation_start).total_seconds() / 60.0)
                    for e in completed_query
                ]
                last_five_df = pd.DataFrame(
                    {"service_time_min": np.array(all_durations[-5:], dtype=np.float32)}
                )
                # Match notebook Cell 75: X_history uses service_time_scaled!
                history_scaled = target_scaler.transform(last_five_df).reshape(1, 5, 1)

                ref_dt = datetime.combine(
                    appointment_date or date.today(),
                    appointment_time or datetime.now().time()
                )
                month_name = ref_dt.strftime("%B")
                day_name = ref_dt.strftime("%A")
                is_morning = ref_dt.hour < 12

                current_dict = {col: 0.0 for col in feature_cols}
                current_dict["WorkingDay"] = 1.0 if ref_dt.weekday() < 5 else 0.0
                if f"Month_{month_name}" in current_dict:
                    current_dict[f"Month_{month_name}"] = 1.0
                if f"DayOfWeek_{day_name}" in current_dict:
                    current_dict[f"DayOfWeek_{day_name}"] = 1.0
                current_dict["AM_PM_morning"] = 1.0 if is_morning else 0.0
                current_dict["AM_PM_afternoon"] = 0.0 if is_morning else 1.0
                current_dict[f"Gender_{gender}"] = 1.0
                current_dict[f"Address_{address}"] = 1.0

                prior_durations = all_durations  # all prior completed consultations in session
                current_dict["previous_session_avg"] = float(np.mean(prior_durations))
                current_dict["previous_session_std"] = float(np.std(prior_durations, ddof=1)) if len(prior_durations) > 1 else 0.0
                current_dict["previous_session_count"] = float(len(prior_durations))

                patient_vec = np.array([[current_dict[col] for col in feature_cols]], dtype=np.float32)

                pred_scaled = model.predict([history_scaled, patient_vec], verbose=0)
                pred_unscaled = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))
                pred_dur = float(pred_unscaled[0][0])
                if np.isfinite(pred_dur):
                    return max(2.0, round(pred_dur, 1)), True, "HanguData LSTM (Reference Only; Test R² = 0.0197)"
            except Exception as ex:
                print("LSTM Service duration inference exception:", ex)

    # Historical reference estimate when 5 completed sessions or demographics are unavailable
    if len(completed_query) > 0:
        hist_durs = [
            max(1.0, (e.consultation_end - e.consultation_start).total_seconds() / 60.0)
            for e in completed_query
        ]
        return round(sum(hist_durs) / len(hist_durs), 1), False, "Historical Reference Estimate (Department Completed Mean)"

    baseline_service = DEPT_SERVICE_BASELINES.get(department_id, 10.0)
    return baseline_service, False, "Historical Reference Estimate (Clinical Baseline)"


# ============================================================
# 6. AUTHENTICATION & ACCESS DECORATORS
# ============================================================

def patient_required(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if "patient_id" not in session:
            flash("Please log in to access your patient portal.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


def staff_required(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if "staff_id" not in session:
            flash("Staff authorization required. Please log in.", "warning")
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated_function


# ============================================================
# 7. PUBLIC & HOME ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/test-db")
def test_db():
    try:
        with db.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return "SQLite Database Connected Successfully!"
    except Exception as e:
        return f"Database Connection Failed: {str(e)}", 500


# ============================================================
# 8. PATIENT REGISTRATION & AUTHENTICATION
# ============================================================

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        phone = request.form.get("phone", "").strip()
        password = request.form.get("password", "")

        if not full_name or not email or not password:
            return render_template("register.html", error="Please fill in all required fields."), 400

        if len(password) < 8:
            return render_template("register.html", error="Password must contain at least 8 characters."), 400

        existing_patient = Patient.query.filter_by(email=email).first()
        if existing_patient:
            return render_template("register.html", error="Email already registered. Please login."), 400

        new_patient = Patient(
            full_name=full_name,
            email=email,
            phone=phone or None,
            password_hash=generate_password_hash(password)
        )

        try:
            db.session.add(new_patient)
            db.session.commit()
            flash("Registration successful! Please log in with your credentials.", "success")
            return redirect(url_for("login"))
        except IntegrityError:
            db.session.rollback()
            return render_template("register.html", error="Email already registered."), 400
        except Exception as e:
            db.session.rollback()
            return render_template("register.html", error=f"Registration failed: {str(e)}"), 500

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        patient = Patient.query.filter_by(email=email).first()

        if patient and check_password_hash(patient.password_hash, password):
            session.clear()
            session["patient_id"] = patient.patient_id
            session["patient_name"] = patient.full_name
            return redirect(url_for("dashboard"))

        return render_template("login.html", error="Invalid email or password. Please try again."), 401

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ============================================================
# 9. PATIENT DASHBOARD & DETAILS
# ============================================================

@app.route("/dashboard")
@patient_required
def dashboard():
    patient = db.session.get(Patient, session["patient_id"])
    if patient is None:
        session.clear()
        return redirect(url_for("login"))

    query = (
        db.session.query(
            Appointment.appointment_id,
            Appointment.appointment_date,
            Appointment.appointment_time,
            Appointment.appointment_status,
            Department.department_name,
            Doctor.doctor_name,
            Doctor.specialization,
            QueueEntry.token_number,
            QueueEntry.queue_status,
            PredictionRecord.predicted_wait_minutes,
            PredictionRecord.predicted_service_minutes
        )
        .join(Department, Appointment.department_id == Department.department_id)
        .outerjoin(Doctor, Appointment.doctor_id == Doctor.doctor_id)
        .outerjoin(QueueEntry, Appointment.appointment_id == QueueEntry.appointment_id)
        .outerjoin(PredictionRecord, Appointment.appointment_id == PredictionRecord.appointment_id)
        .filter(Appointment.patient_id == patient.patient_id)
        .order_by(Appointment.appointment_date.desc(), Appointment.appointment_id.desc())
    )

    appointments_data = []
    for row in query.all():
        appointments_data.append({
            "appointment_id": row[0],
            "appointment_date": row[1].strftime("%Y-%m-%d") if row[1] else "",
            "appointment_time": row[2].strftime("%H:%M") if row[2] else "",
            "appointment_status": row[3],
            "department_name": row[4],
            "doctor_name": row[5] or "Attending Physician",
            "specialization": row[6] or "General",
            "token_number": row[7],
            "queue_status": row[8] or row[3],
            "predicted_wait_minutes": row[9],
            "predicted_service_minutes": row[10]
        })

    return render_template(
        "dashboard.html",
        patient=patient,
        appointments=appointments_data
    )


# ============================================================
# 10. APPOINTMENT BOOKING & INDEPENDENT TOKEN GENERATION
# ============================================================

def _render_appointment_form(error=None, status_code=200):
    departments = Department.query.order_by(Department.department_name).all()
    doctors = Doctor.query.filter_by(availability_status="Available").order_by(Doctor.doctor_name).all()
    return render_template(
        "appointment.html",
        departments=departments,
        doctors=doctors,
        today=date.today().isoformat(),
        error=error
    ), status_code


@app.route("/appointment")
@patient_required
def appointment():
    return _render_appointment_form()


@app.route("/book-appointment", methods=["POST"])
@patient_required
def book_appointment():
    try:
        department_id = int(request.form.get("department_id", ""))
        doctor_id = int(request.form.get("doctor_id", ""))
        appointment_date_str = request.form.get("appointment_date", "")
        appointment_time_str = request.form.get("appointment_time", "")

        selected_date = datetime.strptime(appointment_date_str, "%Y-%m-%d").date()
        selected_time = datetime.strptime(appointment_time_str, "%H:%M").time()

        if selected_date < date.today():
            return _render_appointment_form("Please select today or a future date.", 400)

        department = db.session.get(Department, department_id)
        if department is None:
            return _render_appointment_form("Invalid department selected.", 400)

        doctor = db.session.get(Doctor, doctor_id)
        if doctor is None:
            return _render_appointment_form("Invalid doctor selected.", 400)

        if doctor.department_id != department_id:
            return _render_appointment_form("Selected doctor does not belong to this department.", 400)

        if doctor.availability_status != "Available":
            return _render_appointment_form("Selected doctor is currently unavailable.", 400)

        # Create appointment
        new_appointment = Appointment(
            patient_id=session["patient_id"],
            department_id=department_id,
            doctor_id=doctor_id,
            appointment_date=selected_date,
            appointment_time=selected_time,
            appointment_status="Booked"
        )
        db.session.add(new_appointment)
        db.session.flush()

        # Department-wise and date-wise token numbering (preserved working logic)
        last_token = (
            db.session.query(db.func.max(QueueEntry.token_number))
            .join(Appointment, QueueEntry.appointment_id == Appointment.appointment_id)
            .filter(
                Appointment.department_id == department_id,
                Appointment.appointment_date == selected_date
            )
            .scalar()
        )
        token_number = (last_token or 0) + 1

        new_queue_entry = QueueEntry(
            appointment_id=new_appointment.appointment_id,
            token_number=token_number,
            queue_status="Booked"
        )
        db.session.add(new_queue_entry)

        patients_ahead = (
            db.session.query(db.func.count(QueueEntry.queue_id))
            .join(Appointment, QueueEntry.appointment_id == Appointment.appointment_id)
            .filter(
                Appointment.department_id == department_id,
                Appointment.appointment_date == selected_date,
                QueueEntry.token_number < token_number,
                QueueEntry.queue_status.notin_(["Completed", "Cancelled"])
            )
            .scalar()
        ) or 0

        wait_min, is_lstm_wait, wait_label = predict_patient_wait_time(
            department_id, selected_date, patients_ahead
        )
        service_min, is_lstm_service, service_label = predict_patient_service_duration(
            department_id, selected_date, selected_time
        )

        sched_dt = datetime.combine(selected_date, selected_time)
        pred_consult_dt = sched_dt + timedelta(minutes=wait_min)

        pred_record = PredictionRecord(
            appointment_id=new_appointment.appointment_id,
            predicted_wait_minutes=wait_min,
            predicted_service_minutes=service_min,
            predicted_consultation_time=pred_consult_dt
        )
        db.session.add(pred_record)
        db.session.commit()

        return render_template(
            "booking_success.html",
            token_number=token_number,
            department=department,
            doctor=doctor,
            appointment=new_appointment,
            wait_minutes=wait_min,
            service_minutes=service_min,
            wait_label=wait_label,
            service_label=service_label
        )

    except ValueError:
        db.session.rollback()
        return _render_appointment_form("Please enter valid appointment details.", 400)
    except Exception as e:
        db.session.rollback()
        return _render_appointment_form(f"Booking failed: {str(e)}", 500)


# ============================================================
# 11. PATIENT QUEUE TRACKING PAGE
# ============================================================

def _get_patient_appointments_list(patient_id):
    return (
        db.session.query(Appointment, Department, QueueEntry)
        .join(Department, Appointment.department_id == Department.department_id)
        .outerjoin(QueueEntry, Appointment.appointment_id == QueueEntry.appointment_id)
        .filter(Appointment.patient_id == patient_id)
        .order_by(Appointment.appointment_date.desc(), Appointment.appointment_id.desc())
        .all()
    )


def _count_patients_ahead(appointment, current_token, queue_entry):
    if not queue_entry or queue_entry.queue_status in ["Completed", "Cancelled"]:
        return 0
    return (
        db.session.query(db.func.count(QueueEntry.queue_id))
        .join(Appointment, QueueEntry.appointment_id == Appointment.appointment_id)
        .filter(
            Appointment.department_id == appointment.department_id,
            Appointment.appointment_date == appointment.appointment_date,
            QueueEntry.token_number < current_token,
            QueueEntry.queue_status.notin_(["Completed", "Cancelled"])
        )
        .scalar()
    ) or 0


@app.route("/queue")
@patient_required
def queue():
    patient_id = session["patient_id"]
    req_apt_id = request.args.get("appointment_id", type=int)

    all_apts = _get_patient_appointments_list(patient_id)
    if req_apt_id:
        appointment = Appointment.query.filter_by(
            appointment_id=req_apt_id,
            patient_id=patient_id
        ).first()
    else:
        appointment = all_apts[0][0] if all_apts else None

    if not appointment:
        return render_template("queue.html", appointment=None, all_appointments=[])

    queue_entry = QueueEntry.query.filter_by(appointment_id=appointment.appointment_id).first()
    department = db.session.get(Department, appointment.department_id)
    doctor = db.session.get(Doctor, appointment.doctor_id) if appointment.doctor_id else None

    current_token = queue_entry.token_number if queue_entry else 0
    patients_ahead = _count_patients_ahead(appointment, current_token, queue_entry)

    wait_min, is_lstm_wait, wait_label = predict_patient_wait_time(
        appointment.department_id, appointment.appointment_date, patients_ahead
    )
    service_min, is_lstm_service, service_label = predict_patient_service_duration(
        appointment.department_id, appointment.appointment_date, appointment.appointment_time
    )

    pred_rec = PredictionRecord.query.filter_by(appointment_id=appointment.appointment_id).first()
    if pred_rec is None:
        pred_rec = PredictionRecord(appointment_id=appointment.appointment_id)
        db.session.add(pred_rec)

    sched_time = appointment.appointment_time or time(9, 0)
    sched_dt = datetime.combine(appointment.appointment_date, sched_time)
    pred_rec.predicted_wait_minutes = wait_min
    pred_rec.predicted_service_minutes = service_min
    pred_rec.predicted_consultation_time = sched_dt + timedelta(minutes=wait_min)
    try:
        db.session.commit()
    except Exception:
        db.session.rollback()

    return render_template(
        "queue.html",
        appointment=appointment,
        all_appointments=all_apts,
        queue_entry=queue_entry,
        department=department,
        doctor=doctor,
        token_number=current_token,
        queue_status=queue_entry.queue_status if queue_entry else appointment.appointment_status,
        patients_ahead=patients_ahead,
        wait_minutes=wait_min,
        service_minutes=service_min,
        is_lstm_wait=is_lstm_wait,
        is_lstm_service=is_lstm_service,
        wait_label=wait_label,
        service_label=service_label
    )


# ============================================================
# 12. PATIENT AI PREDICTIONS, HISTORY & DEPARTURE PLANNER
# ============================================================

@app.route("/prediction")
@patient_required
def prediction():
    patient_id = session["patient_id"]
    req_apt_id = request.args.get("appointment_id", type=int)
    travel_min = request.args.get("travel_min", default=20, type=int)
    if travel_min is None or travel_min < 0:
        travel_min = 20

    gender = request.args.get("gender", "").strip()
    address = request.args.get("address", "").strip()

    all_apts = _get_patient_appointments_list(patient_id)
    if req_apt_id:
        appointment = Appointment.query.filter_by(
            appointment_id=req_apt_id,
            patient_id=patient_id
        ).first()
    else:
        appointment = all_apts[0][0] if all_apts else None

    if not appointment:
        return render_template("prediction.html", appointment=None, all_appointments=[], prediction_history=[])

    queue_entry = QueueEntry.query.filter_by(appointment_id=appointment.appointment_id).first()
    department = db.session.get(Department, appointment.department_id)
    doctor = db.session.get(Doctor, appointment.doctor_id) if appointment.doctor_id else None

    current_token = queue_entry.token_number if queue_entry else 0
    patients_ahead = _count_patients_ahead(appointment, current_token, queue_entry)

    wait_min, is_lstm_wait, wait_label = predict_patient_wait_time(
        appointment.department_id, appointment.appointment_date, patients_ahead
    )
    service_min, is_lstm_service, service_label = predict_patient_service_duration(
        appointment.department_id,
        appointment.appointment_date,
        appointment.appointment_time,
        gender=gender or None,
        address=address or None
    )

    sched_time = appointment.appointment_time or time(9, 0)
    sched_dt = datetime.combine(appointment.appointment_date, sched_time)
    consultation_start_est = sched_dt + timedelta(minutes=wait_min)
    consultation_end_est = consultation_start_est + timedelta(minutes=service_min)
    departure_est = consultation_start_est - timedelta(minutes=travel_min + 10)

    pred_rec = PredictionRecord.query.filter_by(appointment_id=appointment.appointment_id).first()
    if pred_rec is None:
        pred_rec = PredictionRecord(appointment_id=appointment.appointment_id)
        db.session.add(pred_rec)

    pred_rec.predicted_wait_minutes = wait_min
    pred_rec.predicted_service_minutes = service_min
    pred_rec.predicted_consultation_time = consultation_start_est
    pred_rec.predicted_at = datetime.now()
    try:
        db.session.commit()
    except Exception:
        db.session.rollback()

    # Load patient's prediction history from predictions table
    history_rows = (
        db.session.query(
            PredictionRecord.prediction_id,
            PredictionRecord.appointment_id,
            PredictionRecord.predicted_wait_minutes,
            PredictionRecord.predicted_service_minutes,
            PredictionRecord.predicted_consultation_time,
            PredictionRecord.predicted_at,
            Department.department_name,
            Doctor.doctor_name,
            Appointment.appointment_date,
            QueueEntry.token_number
        )
        .join(Appointment, PredictionRecord.appointment_id == Appointment.appointment_id)
        .join(Department, Appointment.department_id == Department.department_id)
        .outerjoin(Doctor, Appointment.doctor_id == Doctor.doctor_id)
        .outerjoin(QueueEntry, Appointment.appointment_id == QueueEntry.appointment_id)
        .filter(Appointment.patient_id == patient_id)
        .order_by(PredictionRecord.predicted_at.desc(), PredictionRecord.prediction_id.desc())
        .all()
    )

    prediction_history = [
        {
            "prediction_id": r[0],
            "appointment_id": r[1],
            "wait_minutes": r[2],
            "service_minutes": r[3],
            "consultation_time": r[4].strftime("%Y-%m-%d %I:%M %p") if r[4] else "--",
            "predicted_at": r[5].strftime("%Y-%m-%d %I:%M %p") if r[5] else "--",
            "department_name": r[6],
            "doctor_name": r[7] or "Attending Physician",
            "appointment_date": r[8].strftime("%Y-%m-%d") if r[8] else "--",
            "token_number": r[9]
        }
        for r in history_rows
    ]

    return render_template(
        "prediction.html",
        appointment=appointment,
        all_appointments=all_apts,
        queue_entry=queue_entry,
        department=department,
        doctor=doctor,
        patients_ahead=patients_ahead,
        wait_minutes=wait_min,
        service_minutes=service_min,
        is_lstm_wait=is_lstm_wait,
        is_lstm_service=is_lstm_service,
        wait_label=wait_label,
        service_label=service_label,
        consultation_time=consultation_start_est.strftime("%I:%M %p (%d %b)"),
        consultation_end=consultation_end_est.strftime("%I:%M %p (%d %b)"),
        departure_time=departure_est.strftime("%I:%M %p (%d %b)"),
        travel_min=travel_min,
        selected_gender=gender,
        selected_address=address,
        prediction_history=prediction_history
    )


# ============================================================
# 13. HOSPITAL STAFF SETUP & AUTHENTICATION
# ============================================================

@app.route("/setup-staff", methods=["GET", "POST"])
def setup_staff():
    staff_count = HospitalStaff.query.count()
    is_admin_session = session.get("staff_id") is not None and session.get("staff_role") == "Admin"
    env_passcode = os.environ.get("STAFF_SETUP_PASSCODE")

    # Once the initial staff account exists, protect/disable public setup unless
    # an authenticated Admin is logged in (or STAFF_SETUP_PASSCODE env var is configured).
    setup_locked = staff_count > 0 and not is_admin_session and not env_passcode

    if setup_locked:
        return render_template(
            "create_staff.html",
            staff_count=staff_count,
            setup_locked=True,
            is_admin_session=is_admin_session,
            error="Initial staff setup is already completed and protected. Please log in as an Administrator to create additional staff accounts."
        ), 403

    if request.method == "POST":
        staff_name = request.form.get("staff_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        role = request.form.get("role", "Staff").strip()
        password = request.form.get("password", "")
        setup_passcode = request.form.get("setup_passcode", "")

        if staff_count > 0 and not is_admin_session:
            if not env_passcode or setup_passcode != env_passcode:
                return render_template(
                    "create_staff.html",
                    staff_count=staff_count,
                    setup_locked=False,
                    is_admin_session=is_admin_session,
                    requires_passcode=bool(env_passcode),
                    error="Invalid authorization passcode. Account creation denied."
                ), 403

        if not staff_name or not email or not password:
            return render_template(
                "create_staff.html",
                staff_count=staff_count,
                setup_locked=False,
                is_admin_session=is_admin_session,
                requires_passcode=(staff_count > 0 and not is_admin_session and bool(env_passcode)),
                error="All fields are required."
            ), 400

        if len(password) < 8:
            return render_template(
                "create_staff.html",
                staff_count=staff_count,
                setup_locked=False,
                is_admin_session=is_admin_session,
                requires_passcode=(staff_count > 0 and not is_admin_session and bool(env_passcode)),
                error="Password must be at least 8 characters."
            ), 400

        if HospitalStaff.query.filter_by(email=email).first():
            return render_template(
                "create_staff.html",
                staff_count=staff_count,
                setup_locked=False,
                is_admin_session=is_admin_session,
                requires_passcode=(staff_count > 0 and not is_admin_session and bool(env_passcode)),
                error="Staff email already registered."
            ), 400

        # Ensure initial setup account gets Admin role if first account
        assigned_role = "Admin" if staff_count == 0 else role

        new_staff = HospitalStaff(
            staff_name=staff_name,
            email=email,
            role=assigned_role,
            password_hash=generate_password_hash(password)
        )
        try:
            db.session.add(new_staff)
            db.session.commit()
            return render_template(
                "create_staff.html",
                staff_count=HospitalStaff.query.count(),
                setup_locked=False,
                is_admin_session=is_admin_session,
                success=f"Staff account '{staff_name}' ({assigned_role}) created successfully! You may now log in."
            )
        except Exception as e:
            db.session.rollback()
            return render_template(
                "create_staff.html",
                staff_count=staff_count,
                setup_locked=False,
                is_admin_session=is_admin_session,
                error=f"Database error: {str(e)}"
            ), 500

    return render_template(
        "create_staff.html",
        staff_count=staff_count,
        setup_locked=False,
        is_admin_session=is_admin_session,
        requires_passcode=(staff_count > 0 and not is_admin_session and bool(env_passcode))
    )


@app.route("/admin-login", methods=["GET", "POST"])
def admin_login():
    staff_count = HospitalStaff.query.count()
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        staff = HospitalStaff.query.filter_by(email=email).first()
        if staff and check_password_hash(staff.password_hash, password):
            session.clear()
            session["staff_id"] = staff.staff_id
            session["staff_name"] = staff.staff_name
            session["staff_role"] = staff.role
            return redirect(url_for("admin_dashboard"))

        return render_template(
            "admin_login.html",
            staff_count=staff_count,
            error="Invalid staff credentials. Please check your email and password."
        ), 401

    return render_template("admin_login.html", staff_count=staff_count)


@app.route("/admin-logout")
def admin_logout():
    session.clear()
    return redirect(url_for("admin_login"))


# ============================================================
# 14. ADMIN DASHBOARD
# ============================================================

@app.route("/admin-dashboard")
@staff_required
def admin_dashboard():
    today = date.today()

    total_appointments_today = Appointment.query.filter_by(appointment_date=today).count()

    queue_today_query = (
        db.session.query(QueueEntry.queue_status)
        .join(Appointment, QueueEntry.appointment_id == Appointment.appointment_id)
        .filter(Appointment.appointment_date == today)
    )

    total_queue_today = queue_today_query.count()
    statuses = [r[0] for r in queue_today_query.all()]
    waiting_today = sum(1 for s in statuses if s in ["Waiting", "Reached", "Booked"])
    in_consultation_today = sum(1 for s in statuses if s == "In Consultation")
    completed_today = sum(1 for s in statuses if s == "Completed")

    all_depts = Department.query.order_by(Department.department_id).all()
    dept_stats = []

    for dept in all_depts:
        d_queue = (
            db.session.query(QueueEntry.queue_status)
            .join(Appointment, QueueEntry.appointment_id == Appointment.appointment_id)
            .filter(
                Appointment.department_id == dept.department_id,
                Appointment.appointment_date == today
            )
            .all()
        )
        d_statuses = [q[0] for q in d_queue]
        dept_stats.append({
            "id": dept.department_id,
            "name": dept.department_name,
            "total": len(d_statuses),
            "waiting": sum(1 for s in d_statuses if s in ["Waiting", "Reached", "Booked"]),
            "in_consultation": sum(1 for s in d_statuses if s == "In Consultation"),
            "completed": sum(1 for s in d_statuses if s == "Completed")
        })

    return render_template(
        "admin_dashboard.html",
        today_str=today.strftime("%A, %d %B %Y"),
        total_appointments_today=total_appointments_today,
        total_queue_today=total_queue_today,
        waiting_today=waiting_today,
        in_consultation_today=in_consultation_today,
        completed_today=completed_today,
        dept_stats=dept_stats
    )


# ============================================================
# 15. ADMIN QUEUE MANAGEMENT
# ============================================================

@app.route("/admin-queue")
@staff_required
def admin_queue():
    dept_id = request.args.get("department_id", type=int)
    date_str = request.args.get("appointment_date", "")

    selected_date = date.today()
    if date_str:
        try:
            selected_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            selected_date = date.today()

    departments = Department.query.order_by(Department.department_name).all()

    query = (
        db.session.query(
            QueueEntry.queue_id,
            QueueEntry.token_number,
            QueueEntry.queue_status,
            QueueEntry.arrival_time,
            QueueEntry.consultation_start,
            QueueEntry.consultation_end,
            Patient.full_name,
            Patient.phone,
            Patient.email,
            Department.department_name,
            Doctor.doctor_name,
            Appointment.appointment_date,
            Appointment.appointment_time
        )
        .join(Appointment, QueueEntry.appointment_id == Appointment.appointment_id)
        .join(Patient, Appointment.patient_id == Patient.patient_id)
        .join(Department, Appointment.department_id == Department.department_id)
        .outerjoin(Doctor, Appointment.doctor_id == Doctor.doctor_id)
        .filter(Appointment.appointment_date == selected_date)
    )

    if dept_id:
        query = query.filter(Appointment.department_id == dept_id)

    query = query.order_by(Department.department_name.asc(), QueueEntry.token_number.asc())
    rows = query.all()

    queue_items = []
    reached_count = 0
    waiting_count = 0
    completed_count = 0

    for r in rows:
        status = r[2]
        if status == "Reached":
            reached_count += 1
        elif status == "Waiting":
            waiting_count += 1
        elif status == "Completed":
            completed_count += 1

        queue_items.append({
            "queue_id": r[0],
            "token_number": r[1],
            "queue_status": r[2],
            "arrival_time": r[3].strftime("%I:%M %p") if r[3] else None,
            "consultation_start": r[4].strftime("%I:%M %p") if r[4] else None,
            "consultation_end": r[5].strftime("%I:%M %p") if r[5] else None,
            "patient_name": r[6],
            "patient_phone": r[7],
            "patient_email": r[8],
            "department_name": r[9],
            "doctor_name": r[10] or "Attending Physician",
            "appointment_date": r[11].strftime("%Y-%m-%d"),
            "appointment_time": r[12].strftime("%I:%M %p") if r[12] else None
        })

    return render_template(
        "manage_queue.html",
        departments=departments,
        selected_department_id=dept_id,
        selected_date=selected_date.strftime("%Y-%m-%d"),
        queue_items=queue_items,
        total_queue=len(queue_items),
        reached_count=reached_count,
        waiting_count=waiting_count,
        completed_count=completed_count
    )


@app.route("/admin-update-queue", methods=["POST"])
@staff_required
def admin_update_queue():
    queue_id = request.form.get("queue_id", type=int)
    new_status = request.form.get("queue_status", "").strip()
    selected_dept = request.form.get("department_id", "")
    selected_date = request.form.get("appointment_date", "")

    params = {}
    if selected_dept:
        params["department_id"] = selected_dept
    if selected_date:
        params["appointment_date"] = selected_date

    valid_statuses = {"Booked", "Reached", "Waiting", "In Consultation", "Completed", "Cancelled"}
    if not queue_id or new_status not in valid_statuses:
        params["error"] = "Invalid queue status parameters."
        return redirect(url_for("admin_queue", **params))

    queue_entry = db.session.get(QueueEntry, queue_id)
    if not queue_entry:
        params["error"] = "Queue record not found."
        return redirect(url_for("admin_queue", **params))

    current_status = queue_entry.queue_status or "Booked"
    allowed_next = ALLOWED_STATUS_TRANSITIONS.get(current_status, valid_statuses)
    if new_status not in allowed_next:
        params["error"] = f"Invalid status transition: cannot move from '{current_status}' to '{new_status}'."
        return redirect(url_for("admin_queue", **params))

    now = datetime.now()
    queue_entry.queue_status = new_status

    if new_status in ["Reached", "Waiting"]:
        if not queue_entry.arrival_time:
            queue_entry.arrival_time = now
    elif new_status == "In Consultation":
        if not queue_entry.arrival_time:
            queue_entry.arrival_time = now
        if not queue_entry.consultation_start:
            queue_entry.consultation_start = now
    elif new_status == "Completed":
        if not queue_entry.arrival_time:
            queue_entry.arrival_time = now
        if not queue_entry.consultation_start:
            queue_entry.consultation_start = now
        if not queue_entry.consultation_end:
            queue_entry.consultation_end = now

    appointment = db.session.get(Appointment, queue_entry.appointment_id)
    if appointment:
        appointment.appointment_status = new_status

    try:
        db.session.commit()
        params["success"] = f"Token #{queue_entry.token_number} status updated to '{new_status}'."
    except Exception as e:
        db.session.rollback()
        params["error"] = f"Database update error: {str(e)}"

    return redirect(url_for("admin_queue", **params))


# ============================================================
# 16. DOCTOR MANAGEMENT
# ============================================================

@app.route("/admin-doctors")
@staff_required
def admin_doctors():
    departments = Department.query.order_by(Department.department_name).all()

    doctors_query = (
        db.session.query(
            Doctor.doctor_id,
            Doctor.doctor_name,
            Doctor.department_id,
            Doctor.specialization,
            Doctor.availability_status,
            Department.department_name
        )
        .join(Department, Doctor.department_id == Department.department_id)
        .order_by(Doctor.department_id.asc(), Doctor.doctor_name.asc())
        .all()
    )

    doctors = []
    available_count = 0
    unavailable_count = 0

    for d in doctors_query:
        if d[4] == "Available":
            available_count += 1
        else:
            unavailable_count += 1

        doctors.append({
            "doctor_id": d[0],
            "doctor_name": d[1],
            "department_id": d[2],
            "specialization": d[3],
            "availability_status": d[4],
            "department_name": d[5]
        })

    return render_template(
        "doctor_management.html",
        departments=departments,
        doctors=doctors,
        total_doctors=len(doctors),
        available_doctors=available_count,
        unavailable_doctors=unavailable_count
    )


@app.route("/admin-add-doctor", methods=["POST"])
@staff_required
def admin_add_doctor():
    doctor_name = request.form.get("doctor_name", "").strip()
    department_id = request.form.get("department_id", type=int)
    specialization = request.form.get("specialization", "").strip()
    availability = request.form.get("availability_status", "Available").strip()

    if not doctor_name or not department_id:
        return redirect(url_for("admin_doctors", error="Doctor name and department are required."))

    if availability not in ["Available", "Unavailable"]:
        return redirect(url_for("admin_doctors", error="Invalid availability status."))

    department = db.session.get(Department, department_id)
    if not department:
        return redirect(url_for("admin_doctors", error="Invalid department selected."))

    new_doc = Doctor(
        doctor_name=doctor_name,
        department_id=department_id,
        specialization=specialization,
        availability_status=availability
    )
    try:
        db.session.add(new_doc)
        db.session.commit()
        return redirect(url_for("admin_doctors", success=f"Doctor '{doctor_name}' registered successfully."))
    except Exception as e:
        db.session.rollback()
        return redirect(url_for("admin_doctors", error=f"Failed to add doctor: {str(e)}"))


@app.route("/admin-update-doctor", methods=["POST"])
@staff_required
def admin_update_doctor():
    doctor_id = request.form.get("doctor_id", type=int)
    availability = request.form.get("availability_status", "").strip()

    if not doctor_id or availability not in ["Available", "Unavailable"]:
        return redirect(url_for("admin_doctors", error="Invalid doctor update request."))

    doctor = db.session.get(Doctor, doctor_id)
    if not doctor:
        return redirect(url_for("admin_doctors", error="Doctor record not found."))

    doctor.availability_status = availability
    try:
        db.session.commit()
        return redirect(url_for("admin_doctors", success=f"Updated status for {doctor.doctor_name} to '{availability}'."))
    except Exception as e:
        db.session.rollback()
        return redirect(url_for("admin_doctors", error=f"Database error: {str(e)}"))


# ============================================================
# 17. RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(debug=True, port=5000)
