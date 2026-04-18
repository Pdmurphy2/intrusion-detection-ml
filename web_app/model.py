import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "final_model.joblib"

RAW_FEATURE_COLUMNS = [
    "network_packet_size",
    "protocol_type",
    "login_attempts",
    "session_duration",
    "encryption_used",
    "ip_reputation_score",
    "failed_logins",
    "browser_type",
    "unusual_time_access",
]

ENGINEERED_FEATURE_COLUMNS = [
    "failed_login_reputation_score",
    "login_attempt_reputation_score",
]

MODEL_INPUT_COLUMNS = RAW_FEATURE_COLUMNS + ENGINEERED_FEATURE_COLUMNS

PROTOCOL_TYPES = {"TCP", "UDP", "ICMP"}
ENCRYPTION_TYPES = {"AES", "DES", "None"}
BROWSER_TYPES = {"Chrome", "Firefox", "Edge", "Safari", "Unknown"}
UNUSUAL_TIME_ACCESS = {0, 1}


def load_model():
    """Load the fitted sklearn pipeline (Preprocessor and Model). Preprocessor handles
       scaling and one hot encoding data so model gets the same pipeline of data it was
       trained with"""
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Could not find saved model at {MODEL_PATH}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to load model from {MODEL_PATH}") from exc


def _require_value(raw_data, feature_name):
    """Helper using dict.get(). Checks if a value is returned and raises ValueError if
       no value is found when getting feature data"""
    value = raw_data.get(feature_name)
    if value is None or str(value).strip() == "":
        raise ValueError(f"{feature_name} is required.")
    return value


def prepare_features(raw_data):
    """Convert raw app data into a one row DataFrame the model expects"""
    # Ensure raw data features have a value then cast into a numeric value
    try:
        network_packet_size = int(_require_value(raw_data, "network_packet_size"))
        login_attempts = int(_require_value(raw_data, "login_attempts"))
        session_duration = float(_require_value(raw_data, "session_duration"))
        ip_reputation_score = float(_require_value(raw_data, "ip_reputation_score"))
        failed_logins = int(_require_value(raw_data, "failed_logins"))
        unusual_time_access = int(_require_value(raw_data, "unusual_time_access"))
    except ValueError as err:
        raise ValueError(f"Invalid numeric input: {err}") from err

    # Ensure catigorical data has values and remove any leading or trailing whitespace
    protocol_type = str(_require_value(raw_data, "protocol_type")).strip()
    encryption_used = str(_require_value(raw_data, "encryption_used")).strip()
    browser_type = str(_require_value(raw_data, "browser_type")).strip()

    # Ensure fields are expected values and raise errors if any unexpected values
    if protocol_type not in PROTOCOL_TYPES:
        raise ValueError(f"Invalid protocol_type: {protocol_type}")
    if encryption_used not in ENCRYPTION_TYPES:
        raise ValueError(f"Invalid encryption_used: {encryption_used}")
    if browser_type not in BROWSER_TYPES:
        raise ValueError(f"Invalid browser_type: {browser_type}")
    if network_packet_size <= 0:
        raise ValueError("network_packet_size must be greater than 0.")
    if login_attempts <= 0:
        raise ValueError("login_attempts must be greater than 0.")
    if session_duration <= 0:
        raise ValueError("session_duration must be greater than 0.")
    if failed_logins < 0:
        raise ValueError("failed_logins must be 0 or greater.")
    if not 0.0 <= ip_reputation_score <= 1.0:
        raise ValueError("ip_reputation_score must be between 0 and 1.")
    if unusual_time_access not in UNUSUAL_TIME_ACCESS:
        raise ValueError("unusual_time_access must be 0 or 1.")

    # Create a dict that will later be a df row
    row = {
        "network_packet_size": network_packet_size,
        "protocol_type": protocol_type,
        "login_attempts": login_attempts,
        "session_duration": session_duration,
        "encryption_used": encryption_used,
        "ip_reputation_score": ip_reputation_score,
        "failed_logins": failed_logins,
        "browser_type": browser_type,
        "unusual_time_access": unusual_time_access,
    }

    # Recreate the engineered features the deployed model was trained on.
    row["failed_login_reputation_score"] = row["ip_reputation_score"] * (row["failed_logins"] + 1)
    row["login_attempt_reputation_score"] = row["login_attempts"] * row["ip_reputation_score"]

    return pd.DataFrame([row], columns=MODEL_INPUT_COLUMNS)


def predict_attack(model, raw_data):
    """Have fitted model predict if network session is an attack using raw data from app. Uses
       helper functions prepare_features() to prepare the data. Return dict of prediction details"""
    
    features = prepare_features(raw_data)
    prediction = int(model.predict(features)[0])
    normal_probability = float(model.predict_proba(features)[0, 0])
    attack_probability = float(model.predict_proba(features)[0, 1])

    return {
        "prediction": prediction,
        "label": "Attack" if prediction == 1 else "Normal",
        "attack_probability": attack_probability,
        "normal_probability": normal_probability,
    }
