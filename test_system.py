import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from datetime import date, datetime, time, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
APP_DIR = PROJECT_ROOT / "App"
ORIG_DB = PROJECT_ROOT / "database" / "opd_management.db"

# Create a temporary copy of the existing SQLite database so tests never alter real data
_TMP_BASE = os.environ.get("OPD_TEST_TMPDIR")
_TEMP_DIR = tempfile.TemporaryDirectory(dir=_TMP_BASE)
TEST_DB = Path(_TEMP_DIR.name) / "test_opd_management.db"
shutil.copy2(ORIG_DB, TEST_DB)

os.environ["OPD_DATABASE_PATH"] = str(TEST_DB)
sys.path.insert(0, str(APP_DIR))
import app as flask_app_module

flask_app_module.app.config["TESTING"] = True

from app import (
    app,
    db,
    Patient,
    Department,
    Doctor,
    Appointment,
    QueueEntry,
    PredictionRecord,
    HospitalStaff,
    predict_patient_wait_time,
    predict_patient_service_duration,
)


class SmartOPDComprehensiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    @classmethod
    def tearDownClass(cls):
        with app.app_context():
            db.session.remove()
            db.engine.dispose()
        try:
            _TEMP_DIR.cleanup()
        except Exception:
            pass

    def test_01_homepage_and_db_connectivity(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Smart", resp.data)
        self.assertIn(b"Staff Portal", resp.data)

        db_resp = self.client.get("/test-db")
        self.assertEqual(db_resp.status_code, 200)
        self.assertIn(b"Connected Successfully", db_resp.data)
        print("[PASS] 1. Home page & DB connectivity load cleanly")

    def test_02_existing_data_intact_in_original_db(self):
        conn = sqlite3.connect(f"file:{ORIG_DB.as_posix()}?mode=ro", uri=True)
        cur = conn.cursor()
        p_cnt = cur.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        d_cnt = cur.execute("SELECT COUNT(*) FROM departments").fetchone()[0]
        doc_cnt = cur.execute("SELECT COUNT(*) FROM doctors").fetchone()[0]
        a_cnt = cur.execute("SELECT COUNT(*) FROM appointments").fetchone()[0]
        q_cnt = cur.execute("SELECT COUNT(*) FROM queue").fetchone()[0]
        conn.close()
        self.assertGreaterEqual(p_cnt, 4)
        self.assertEqual(d_cnt, 6)
        self.assertGreaterEqual(doc_cnt, 6)
        self.assertGreaterEqual(a_cnt, 4)
        self.assertGreaterEqual(q_cnt, 3)
        print(f"[PASS] 20. Original DB intact: {p_cnt} patients, {d_cnt} depts, {doc_cnt} doctors, {a_cnt} appointments, {q_cnt} queue")

    def test_03_patient_registration_login_and_dashboard(self):
        # Unauthenticated dashboard redirects
        unauth = self.client.get("/dashboard")
        self.assertEqual(unauth.status_code, 302)

        # Short password rejected
        bad_reg = self.client.post("/register", data={
            "full_name": "Short Pass",
            "email": "short@example.com",
            "password": "123"
        })
        self.assertEqual(bad_reg.status_code, 400)

        email = "integration_patient@hospital.org"
        reg_resp = self.client.post("/register", data={
            "full_name": "Integration Test Patient",
            "email": email,
            "phone": "9876543210",
            "password": "SecurePassword123"
        }, follow_redirects=True)
        self.assertEqual(reg_resp.status_code, 200)
        self.assertIn(b"Patient Login", reg_resp.data)

        # Wrong password rejected
        bad_login = self.client.post("/login", data={"email": email, "password": "wrong"})
        self.assertEqual(bad_login.status_code, 401)

        # Valid login -> dashboard
        login_resp = self.client.post("/login", data={
            "email": email,
            "password": "SecurePassword123"
        }, follow_redirects=True)
        self.assertEqual(login_resp.status_code, 200)
        self.assertIn(b"Welcome, Integration Test Patient", login_resp.data)
        self.assertIn(b"Book Appointment", login_resp.data)
        self.assertIn(b"View My Queue", login_resp.data)
        self.assertIn(b"View Predictions", login_resp.data)
        print("[PASS] 2, 3, 4. Patient registration, login, route protection & dashboard work")

    def test_04_token_independence_and_date_reset_and_booking_confirmation(self):
        with self.client.session_transaction() as sess:
            sess["patient_id"] = 3
            sess["patient_name"] = "Patient 3"

        day_one = date.today() + timedelta(days=10)
        day_two = date.today() + timedelta(days=11)

        # Book 2 appointments in Cardiology (dept 2, doc 2) on day_one -> tokens 1, 2
        r_c1 = self.client.post("/book-appointment", data={
            "department_id": 2, "doctor_id": 2,
            "appointment_date": day_one.isoformat(), "appointment_time": "09:30"
        })
        self.assertEqual(r_c1.status_code, 200)
        self.assertIn(b"#1", r_c1.data)
        self.assertIn(b"Cardiology", r_c1.data)

        r_c2 = self.client.post("/book-appointment", data={
            "department_id": 2, "doctor_id": 2,
            "appointment_date": day_one.isoformat(), "appointment_time": "10:00"
        })
        self.assertEqual(r_c2.status_code, 200)
        self.assertIn(b"#2", r_c2.data)

        # Book 1 appointment in General Medicine (dept 1, doc 1) on day_one -> starts independently at #1
        r_g1 = self.client.post("/book-appointment", data={
            "department_id": 1, "doctor_id": 1,
            "appointment_date": day_one.isoformat(), "appointment_time": "09:45"
        })
        self.assertEqual(r_g1.status_code, 200)
        self.assertIn(b"#1", r_g1.data)
        self.assertIn(b"General Medicine", r_g1.data)

        # Book Cardiology on day_two -> resets to #1
        r_c_next = self.client.post("/book-appointment", data={
            "department_id": 2, "doctor_id": 2,
            "appointment_date": day_two.isoformat(), "appointment_time": "09:00"
        })
        self.assertEqual(r_c_next.status_code, 200)
        self.assertIn(b"#1", r_c_next.data)

        # Wrong department-doctor pairing rejected
        r_mismatch = self.client.post("/book-appointment", data={
            "department_id": 1, "doctor_id": 2,
            "appointment_date": day_one.isoformat(), "appointment_time": "11:00"
        })
        self.assertEqual(r_mismatch.status_code, 400)
        print("[PASS] 5, 6, 7, 8, 9. Booking, confirmation, department-wise tokens & date reset work")

    def test_05_queue_page_and_patients_ahead_calculation(self):
        # On day_one Cardiology, tokens 1 and 2 exist; book token 3 for patient 4
        day_three = date.today() + timedelta(days=12)
        with self.client.session_transaction() as sess:
            sess["patient_id"] = 3

        self.client.post("/book-appointment", data={
            "department_id": 3, "doctor_id": 3,
            "appointment_date": day_three.isoformat(), "appointment_time": "09:00"
        })
        self.client.post("/book-appointment", data={
            "department_id": 3, "doctor_id": 3,
            "appointment_date": day_three.isoformat(), "appointment_time": "09:15"
        })

        with self.client.session_transaction() as sess:
            sess["patient_id"] = 4

        self.client.post("/book-appointment", data={
            "department_id": 3, "doctor_id": 3,
            "appointment_date": day_three.isoformat(), "appointment_time": "09:30"
        })

        with app.app_context():
            apts = Appointment.query.filter_by(department_id=3, appointment_date=day_three).order_by(Appointment.appointment_id.asc()).all()
            self.assertEqual(len(apts), 3)
            q1 = QueueEntry.query.filter_by(appointment_id=apts[0].appointment_id).first()
            q2 = QueueEntry.query.filter_by(appointment_id=apts[1].appointment_id).first()
            q3 = QueueEntry.query.filter_by(appointment_id=apts[2].appointment_id).first()
            # Mark token 1 as Completed and token 2 as Cancelled
            q1.queue_status = "Completed"
            q2.queue_status = "Cancelled"
            db.session.commit()
            apt3_id = apts[2].appointment_id

        # Patient 4 views queue for apt3 -> token 3 should have 0 patients ahead!
        resp = self.client.get(f"/queue?appointment_id={apt3_id}")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"#3", resp.data)
        self.assertIn(b"<h3>0</h3>", resp.data)
        print("[PASS] 10, 11. Queue displays token & counts patients ahead excluding Completed/Cancelled")

    def test_06_staff_setup_protection_login_and_admin_dashboard(self):
        # Unauthenticated admin dashboard redirects
        self.assertEqual(self.client.get("/admin-dashboard").status_code, 302)

        # First staff account creation allowed when hospital_staff is empty
        r_setup = self.client.post("/setup-staff", data={
            "staff_name": "Chief Admin",
            "email": "chiefadmin@hospital.org",
            "role": "Admin",
            "password": "AdminPassword123"
        })
        self.assertEqual(r_setup.status_code, 200)
        self.assertIn(b"created successfully", r_setup.data)

        # Public setup route must now be locked/protected when logged out!
        r_locked = self.client.get("/setup-staff")
        self.assertEqual(r_locked.status_code, 403)

        # Staff login works
        r_login = self.client.post("/admin-login", data={
            "email": "chiefadmin@hospital.org",
            "password": "AdminPassword123"
        }, follow_redirects=True)
        self.assertEqual(r_login.status_code, 200)
        self.assertIn(b"Hospital Administration Dashboard", r_login.data)
        self.assertIn(b"General Medicine", r_login.data)
        self.assertIn(b"Cardiology", r_login.data)
        self.assertIn(b"Dermatology", r_login.data)
        self.assertIn(b"Orthopedics", r_login.data)
        self.assertIn(b"Pediatrics", r_login.data)
        self.assertIn(b"Gynecology", r_login.data)
        print("[PASS] 12, 13. Staff setup protection, login, and 6-department admin dashboard work")

    def test_07_admin_queue_updates_timestamps_and_invalid_transitions(self):
        # Log in as staff
        with self.client.session_transaction() as sess:
            sess["staff_id"] = 1
            sess["staff_name"] = "Chief Admin"
            sess["staff_role"] = "Admin"

        with app.app_context():
            q_entry = QueueEntry.query.filter_by(queue_status="Booked").first()
            self.assertIsNotNone(q_entry)
            qid = q_entry.queue_id

        # Booked -> Reached (sets arrival_time)
        self.client.post("/admin-update-queue", data={"queue_id": qid, "queue_status": "Reached"})
        with app.app_context():
            q = db.session.get(QueueEntry, qid)
            self.assertEqual(q.queue_status, "Reached")
            self.assertIsNotNone(q.arrival_time)

        # Reached -> In Consultation (sets consultation_start)
        self.client.post("/admin-update-queue", data={"queue_id": qid, "queue_status": "In Consultation"})
        with app.app_context():
            q = db.session.get(QueueEntry, qid)
            self.assertEqual(q.queue_status, "In Consultation")
            self.assertIsNotNone(q.consultation_start)

        # In Consultation -> Completed (sets consultation_end)
        self.client.post("/admin-update-queue", data={"queue_id": qid, "queue_status": "Completed"})
        with app.app_context():
            q = db.session.get(QueueEntry, qid)
            self.assertEqual(q.queue_status, "Completed")
            self.assertIsNotNone(q.consultation_end)

        # Invalid transition: Completed -> Waiting must be blocked
        r_bad = self.client.post("/admin-update-queue", data={"queue_id": qid, "queue_status": "Waiting"}, follow_redirects=True)
        self.assertIn(b"Invalid status transition", r_bad.data)
        with app.app_context():
            q = db.session.get(QueueEntry, qid)
            self.assertEqual(q.queue_status, "Completed")

        print("[PASS] 14, 15. Staff queue updates, timestamps, and invalid transition blocking verified")

    def test_08_doctor_management_and_unavailable_booking_blocked(self):
        with self.client.session_transaction() as sess:
            sess["staff_id"] = 1
            sess["staff_name"] = "Chief Admin"
            sess["staff_role"] = "Admin"

        # Add a new doctor
        r_add = self.client.post("/admin-add-doctor", data={
            "doctor_name": "Dr. Test Neurologist",
            "department_id": 1,
            "specialization": "Internal Medicine",
            "availability_status": "Available"
        }, follow_redirects=True)
        self.assertEqual(r_add.status_code, 200)
        self.assertIn(b"Dr. Test Neurologist", r_add.data)

        # Mark Doctor 4 (Orthopedics) Unavailable
        self.client.post("/admin-update-doctor", data={"doctor_id": 4, "availability_status": "Unavailable"})

        # Patient tries to view appointment page -> Doctor 4 should not be in dropdown
        with self.client.session_transaction() as sess:
            sess["patient_id"] = 3

        apt_page = self.client.get("/appointment")
        self.assertNotIn(b"Dr. Priya Singh", apt_page.data)

        # Direct POST to book unavailable doctor 4 must be rejected (400)
        book_unavail = self.client.post("/book-appointment", data={
            "department_id": 4,
            "doctor_id": 4,
            "appointment_date": (date.today() + timedelta(days=5)).isoformat(),
            "appointment_time": "11:00"
        })
        self.assertEqual(book_unavail.status_code, 400)
        self.assertIn(b"unavailable", book_unavail.data.lower())
        print("[PASS] 16, 17. Doctor management and unavailable doctor booking block verified")

    def test_09_predictions_fallback_and_live_lstm_inference(self):
        with self.client.session_transaction() as sess:
            sess["patient_id"] = 4

        # Fallback case (insufficient 5-arrival sequence / missing demographics)
        resp = self.client.get("/prediction?travel_min=25")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Queue-Based Estimate", resp.data)
        self.assertIn(b"Historical Reference Estimate", resp.data)
        self.assertIn(b"Saved Prediction History", resp.data)
        self.assertIn(b"User-Provided 25 min travel", resp.data)

        # Now seed 6 complete real sequential arrivals in Dept 5 (Pediatrics) on a test date
        # where earlier patients have started consultations before later patients arrive,
        # and verify BOTH LSTM models run without errors and return valid float predictions!
        test_dt = date.today() + timedelta(days=20)
        base_t = datetime.combine(test_dt, time(9, 0))
        with app.app_context():
            for idx in range(6):
                arr_t = base_t + timedelta(minutes=idx * 10)
                start_t = arr_t + timedelta(minutes=5 + (idx % 4))
                end_t = start_t + timedelta(minutes=8 + (idx % 3))
                a = Appointment(
                    patient_id=3,
                    department_id=5,
                    doctor_id=5,
                    appointment_date=test_dt,
                    appointment_time=(base_t + timedelta(minutes=idx * 10)).time(),
                    appointment_status="Completed"
                )
                db.session.add(a)
                db.session.flush()
                q = QueueEntry(
                    appointment_id=a.appointment_id,
                    token_number=idx + 1,
                    queue_status="Completed",
                    arrival_time=arr_t,
                    consultation_start=start_t,
                    consultation_end=end_t
                )
                db.session.add(q)
            db.session.commit()

            w_min, w_lstm, w_lbl = predict_patient_wait_time(5, test_dt, 2)
            s_min, s_lstm, s_lbl = predict_patient_service_duration(5, test_dt, time(10, 0), gender="F", address="In the city")
            self.assertTrue(w_lstm, f"Expected live WaitData LSTM, got {w_lbl}")
            self.assertTrue(s_lstm, f"Expected live HanguData LSTM, got {s_lbl}")
            self.assertGreater(w_min, 0.0)
            self.assertGreater(s_min, 0.0)

            pred_cnt = PredictionRecord.query.count()
            self.assertGreater(pred_cnt, 0)
            print(f"[PASS] 18, 19. Fallback handling, live WaitData LSTM ({w_min}m), live HanguData LSTM ({s_min}m), & prediction history ({pred_cnt}) verified")


if __name__ == "__main__":
    unittest.main(verbosity=2)
