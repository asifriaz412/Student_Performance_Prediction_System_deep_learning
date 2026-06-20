"""
Flask API for Student Performance Prediction (OULAD)
Serves a web interface and exposes /predict endpoint.
Handles missing files gracefully – will not crash on startup.
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf

# ------------------------- CONFIG -------------------------
MODEL_DIR = "models"
PREPROC_DIR = "preprocessors"
MODEL_EXT = ".keras"

# Auto-detect template folder
if os.path.exists("templates"):
    TEMPLATE_DIR = "templates"
elif os.path.exists("frontend"):
    TEMPLATE_DIR = "frontend"
else:
    TEMPLATE_DIR = None
    print("⚠️ No 'templates' or 'frontend' folder found. HTML page won't be served.")

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# ----------------------- LOAD ASSETS (with error handling) -----------------------
def load_assets():
    assets = {}
    # Load models (skip missing ones)
    model_names = ["ANN", "LSTM", "Bi-LSTM", "GRU", "CNN-LSTM"]
    for name in model_names:
        path = os.path.join(MODEL_DIR, f"{name}{MODEL_EXT}")
        if os.path.exists(path):
            try:
                assets[name] = tf.keras.models.load_model(path)
                print(f"✅ Loaded {name}")
            except Exception as e:
                print(f"⚠️ Error loading {name}: {e}")
        else:
            print(f"⚠️ Model {name} not found at {path}")

    # Try to load preprocessors – if any file missing, skip and print warning
    required_files = [
        "static_scaler.pkl", "demo_scaler.pkl", "label_encoders.pkl",
        "static_feature_cols.pkl", "demo_cols.pkl", "seq_params.pkl"
    ]
    missing = [f for f in required_files if not os.path.exists(os.path.join(PREPROC_DIR, f))]
    if missing:
        print(f"⚠️ Missing preprocessor files: {missing}")
        print("   The /predict endpoint will not work. Frontend will use simulation.")
        assets['preprocessors_loaded'] = False
        return assets

    try:
        with open(os.path.join(PREPROC_DIR, "static_scaler.pkl"), "rb") as f:
            assets['static_scaler'] = pickle.load(f)
        with open(os.path.join(PREPROC_DIR, "demo_scaler.pkl"), "rb") as f:
            assets['demo_scaler'] = pickle.load(f)
        with open(os.path.join(PREPROC_DIR, "label_encoders.pkl"), "rb") as f:
            assets['label_encoders'] = pickle.load(f)
        with open(os.path.join(PREPROC_DIR, "static_feature_cols.pkl"), "rb") as f:
            assets['static_feature_cols'] = pickle.load(f)
        with open(os.path.join(PREPROC_DIR, "demo_cols.pkl"), "rb") as f:
            assets['demo_cols'] = pickle.load(f)
        with open(os.path.join(PREPROC_DIR, "seq_params.pkl"), "rb") as f:
            seq_params = pickle.load(f)
            assets['max_seq_len'] = seq_params['max_seq_len']
            assets['n_features'] = seq_params['n_features']
        assets['preprocessors_loaded'] = True
        print("✅ Preprocessors loaded successfully.")
    except Exception as e:
        print(f"⚠️ Error loading preprocessors: {e}")
        assets['preprocessors_loaded'] = False

    return assets

assets = load_assets()

# ----------------------- HELPER FUNCTIONS -----------------------
def preprocess_static_input(data):
    if not assets.get('preprocessors_loaded'):
        raise RuntimeError("Preprocessors not loaded – cannot preprocess input.")
    demo = data.get("demographics", {})
    extra = {
        "avg_score": data.get("avg_score", 0),
        "submission_count": data.get("submission_count", 0),
        "total_clicks": data.get("total_clicks", 0),
        "active_days": data.get("active_days", 0),
        "engagement_ratio": data.get("engagement_ratio", 0)
    }
    le_dict = assets['label_encoders']
    encoded = {}
    for col in ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']:
        val = demo.get(col, None)
        if val is None or val not in le_dict[col].classes_:
            val = le_dict[col].classes_[0]
        encoded[col] = le_dict[col].transform([val])[0]
    encoded['num_of_prev_attempts'] = float(demo.get('num_of_prev_attempts', 0))
    encoded['studied_credits'] = float(demo.get('studied_credits', 0))
    feature_vec = [encoded[col] if col in encoded else extra[col] for col in assets['static_feature_cols']]
    arr = np.array([feature_vec])
    arr_scaled = assets['static_scaler'].transform(arr)
    return arr_scaled

def preprocess_sequence_input(data):
    if not assets.get('preprocessors_loaded'):
        raise RuntimeError("Preprocessors not loaded.")
    demo = data.get("demographics", {})
    daily = data.get("daily_data", None)
    if not daily:
        raise ValueError("Missing 'daily_data' for sequence models")
    le_dict = assets['label_encoders']
    encoded_demo = {}
    for col in ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']:
        val = demo.get(col, None)
        if val is None or val not in le_dict[col].classes_:
            val = le_dict[col].classes_[0]
        encoded_demo[col] = le_dict[col].transform([val])[0]
    encoded_demo['num_of_prev_attempts'] = float(demo.get('num_of_prev_attempts', 0))
    encoded_demo['studied_credits'] = float(demo.get('studied_credits', 0))
    demo_vec = np.array([encoded_demo[col] for col in assets['demo_cols']], dtype=np.float32)
    demo_num = demo_vec[-2:]
    demo_num_scaled = assets['demo_scaler'].transform(demo_num.reshape(1, -1))[0]
    demo_vec[-2:] = demo_num_scaled

    days = [int(rec.get("day", 0)) for rec in daily]
    min_day, max_day = min(days), max(days)
    day_range = np.arange(min_day, max_day + 1)
    seq_len = len(day_range)
    sum_click_arr = np.zeros(seq_len)
    activity_arr = np.zeros(seq_len)
    score_arr = np.zeros(seq_len)
    daily_dict = {int(rec['day']): rec for rec in daily}
    last_score = 0.0
    for i, day in enumerate(day_range):
        rec = daily_dict.get(day, {})
        sum_click_arr[i] = float(rec.get("sum_click", 0))
        activity_arr[i] = float(rec.get("activity_count", 0))
        score_val = rec.get("score")
        if score_val is not None and score_val != '':
            last_score = float(score_val)
        score_arr[i] = last_score
    time_varying = np.stack([sum_click_arr, activity_arr, score_arr], axis=1)
    demo_repeated = np.tile(demo_vec, (seq_len, 1))
    instance_seq = np.concatenate([time_varying, demo_repeated], axis=1)
    max_len = assets['max_seq_len']
    n_feat = assets['n_features']
    padded = np.zeros((max_len, n_feat))
    padded[:seq_len, :] = instance_seq
    return np.expand_dims(padded, axis=0)

# ----------------------- ROUTES -----------------------
@app.route('/')
def index():
    if TEMPLATE_DIR is None:
        return "Frontend not found. Place index.html in 'templates/' or 'frontend/' folder.", 404
    return render_template('index.html')

@app.route('/categories')
def get_categories():
    if 'label_encoders' not in assets:
        return jsonify({})
    cats = {}
    for col, le in assets['label_encoders'].items():
        cats[col] = list(le.classes_)
    return jsonify(cats)

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

@app.route('/models')
def list_models():
    available = [m for m in ["ANN", "LSTM", "Bi-LSTM", "GRU", "CNN-LSTM"] if m in assets]
    return jsonify({"models": available})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        req = request.get_json(force=True)
        model_name = req.get("model")
        if not model_name:
            return jsonify({"error": "Missing 'model' field"}), 400

        model = assets.get(model_name)
        if not model:
            return jsonify({
                "error": f"Model '{model_name}' not available.",
                "available_models": list_models().get_json()['models']
            }), 404

        if not assets.get('preprocessors_loaded'):
            return jsonify({
                "error": "Preprocessors not loaded. Run main_script.py first.",
                "status": "failed"
            }), 500

        data = req.get("data", {})
        if not data:
            return jsonify({"error": "Missing 'data' field"}), 400

        if model_name == "ANN":
            X = preprocess_static_input(data)
        else:
            X = preprocess_sequence_input(data)

        prob = float(model.predict(X, verbose=0)[0][0])
        pred = 1 if prob >= 0.5 else 0
        return jsonify({
            "model": model_name,
            "prediction": pred,
            "probability": round(prob, 4),
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500

# ------------------------- START -------------------------
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Flask server starting...")
    if TEMPLATE_DIR:
        print(f"📁 Template folder: {TEMPLATE_DIR}/")
    else:
        print("⚠️ No frontend folder – HTML page will not load.")
    print("🔗 Open http://localhost:5000 in your browser.")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)