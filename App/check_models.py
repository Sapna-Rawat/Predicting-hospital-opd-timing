from pathlib import Path
import joblib
from tensorflow.keras.models import load_model

project_folder = Path(__file__).resolve().parent.parent
models_folder = project_folder / "Models"

files = [
    "opd_wait_lstm.keras",
    "feature_scaler.pkl",
    "target_scaler.pkl",
    "selected_features.pkl",
    "hangu_service_lstm.keras",
    "hangu_target_scaler.pkl",
    "hangu_feature_columns.pkl"
]

for file in files:
    path = models_folder / file
    print(file, "->", "FOUND" if path.exists() else "MISSING")

# Load both models
wait_model = load_model(models_folder / "opd_wait_lstm.keras")
service_model = load_model(models_folder / "hangu_service_lstm.keras")

print("\nWait model input:", wait_model.input_shape)
print("Service model input:", service_model.input_shape)

# Verify HanguData scaler
scaler = joblib.load(models_folder / "hangu_target_scaler.pkl")
print("Service scaler features:", scaler.n_features_in_)