from flask import Flask, render_template, request, session, redirect, url_for
import time
from datetime import datetime

from model import load_model, predict_attack

app = Flask(__name__)
app.secret_key = "simple-secret-key"

USERNAME = "admin"
PASSWORD = "password123"

MODEL = load_model()


@app.route("/", methods=["GET", "POST"])
def login():
    if "program_start_time" not in session:
        session["program_start_time"] = time.time()
        session["start_readable"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session["login_attempts"] = 0
        session["failed_attempts"] = 0
        session["logged_in"] = False

    message = ""

    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")

        session["login_attempts"] += 1

        if username == USERNAME and password == PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("home"))
        else:
            session["failed_attempts"] += 1
            message = "Incorrect credentials. Try again."

    return render_template("login.html", message=message)


@app.route("/home")
def home():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("home.html")


@app.route("/logout", methods=["POST"])
def logout():
    if "program_start_time" not in session:
        return redirect(url_for("login"))

    end_time = time.time()
    end_readable = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_time = end_time - session["program_start_time"]

    # Gather extra ML fields from form
    network_packet_size = int(request.form.get("network_packet_size", 0))
    protocol_type = request.form.get("protocol_type", "")
    encryption_used = request.form.get("encryption_used", "")
    ip_reputation_score = float(request.form.get("ip_reputation_score", 0.0))
    ip_reputation_score = max(0.0, min(1.0, ip_reputation_score))
    browser_type = request.form.get("browser_type", "")
    unusual_time_access = int(request.form.get("unusual_time_access", 0))

    ml_data = {
        "network_packet_size": network_packet_size,
        "protocol_type": protocol_type,
        "login_attempts": session.get("login_attempts", 0),
        "session_duration": round(total_time, 2),
        "encryption_used": encryption_used,
        "ip_reputation_score": ip_reputation_score,
        "failed_logins": session.get("failed_attempts", 0),
        "browser_type": browser_type,
        "unusual_time_access": unusual_time_access
    }

    # Handle errors with model predictions
    try:
        prediction_result = predict_attack(MODEL, ml_data)
        prediction_error = ""
    except ValueError as err:
        prediction_result = None
        prediction_error = str(err)
    except Exception:
        prediction_result = None
        prediction_error = "The model could not generate a prediction unexpectedly. Please try again."

    data = {
        "status": "LOGGED OUT",
        "start_time": session["start_readable"],
        "end_time": end_readable,
        "total_time": f"{total_time:.2f}",
        "login_attempts": session.get("login_attempts", 0),
        "failed_attempts": session.get("failed_attempts", 0),
        "ml_data": ml_data,
        "prediction_result": prediction_result,
        "prediction_error": prediction_error,
    }

    session.clear()

    return render_template("result.html", data=data)


if __name__ == "__main__":
    app.run(debug=True)
