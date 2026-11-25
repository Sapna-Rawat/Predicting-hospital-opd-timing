# 🏥 Patient OPD Waiting Time Prediction

An MCA Final Year Project — a Flask web application that predicts OPD waiting times using dual LSTM deep learning models for queue management.

---

## 📋 Features

- **Patient Portal** — Register, login, book appointments, track queue, view AI predictions
- **Staff/Admin Portal** — Manage queue statuses, doctor availability, view analytics dashboard
- **Dual LSTM Models:**
  - `WaitData LSTM` — Predicts patient waiting time (21 features, 5-step sequences)
  - `HanguData LSTM` — Predicts consultation/service duration (28 features)
- **Real-time Queue Tracking** — Live token numbers, position, estimated wait
- **Departure Planner** — Recommends when to leave home based on predicted wait

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.13, Flask 3.x |
| Database | SQLite3 |
| ML Models | TensorFlow / Keras (LSTM) |
| Data Processing | NumPy, Pandas, Scikit-learn, Joblib |
| Frontend | HTML5, CSS3, Bootstrap 5, Jinja2 |

---

## 🚀 Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/patient-opd-waiting-time.git
cd patient-opd-waiting-time
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
cd App
python app.py
```

### 5. Open in browser
```
http://127.0.0.1:5000
```

---

## 👤 First-Time Setup

1. Visit `http://127.0.0.1:5000/setup-staff` to create the **first Admin account**
2. Register as a patient at `/register`
3. Book an appointment at `/appointment`
4. Track queue at `/queue`
5. View AI predictions at `/prediction`

---

## 📁 Project Structure

```
Patient OPD Waiting Time/
├── App/
│   ├── app.py                  # Main Flask application
│   └── templates/              # HTML templates (15 pages)
├── Models/                     # Trained LSTM models (.keras, .pkl)
├── database/
│   └── opd_management.db       # SQLite database
├── Notebooks/                  # Jupyter notebooks (EDA + training)
├── requirements.txt
├── test_system.py              # System test suite (9 groups, all passing)
└── README.md
```

---

## 🧪 Running Tests

```bash
python test_system.py
```
Expected: `Ran 9 tests — OK` ✅

---

## 📊 Model Performance

| Model | Dataset | Architecture | Notes |
|---|---|---|---|
| WaitData LSTM | OPD Wait Records | LSTM (21 features, seq=5) | Live predictions need ≥5 arrivals |
| HanguData LSTM | Consultation Records | LSTM (28 features, seq=5) | R²=0.0197; fallback shown when insufficient data |

---

## 👨‍💻 Author

MCA Final Year Project
