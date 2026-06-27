"""
app.py – Flask prediction server
Adds /shap and /fairness endpoints that serve pre-computed results.
Prediction behaviour is unchanged.

FIXED:
  - All paths resolved relative to this file's directory so the app works
    regardless of the working directory from which it is launched.
  - Graceful /predict when preprocessors are missing (clear error message).
  - /health reports detailed load status.
  - CORS header added so browser-based frontends can call the API.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, make_response
import warnings
warnings.filterwarnings('ignore')

# ── Resolve project root relative to this file ────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "models")
PREPROC_DIR = os.path.join(BASE_DIR, "preprocessors")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOT_DIR    = os.path.join(BASE_DIR, "plots")
DATA_PATH   = os.path.join(BASE_DIR, "data") + os.sep

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,    exist_ok=True)

app = Flask(__name__)

# ── CORS helper ───────────────────────────────────────────────────────────────
def _add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.after_request
def after_request(response):
    return _add_cors(response)

@app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
@app.route("/<path:path>",             methods=["OPTIONS"])
def options_handler(path):
    return _add_cors(make_response("", 204))

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load preprocessors
# ─────────────────────────────────────────────────────────────────────────────
LOAD_ERRORS = []

scaler              = None
label_encoders      = None
static_feature_cols = None
MAX_SEQ_LEN         = None
n_features          = None

print(f"[app.py] BASE_DIR = {BASE_DIR}")
print("Loading preprocessors…")

_preproc_files = {
    "static_scaler.pkl":      "scaler",
    "label_encoders.pkl":     "label_encoders",
    "static_feature_cols.pkl":"static_feature_cols",
    "seq_params.pkl":         "seq_params",
}

_loaded = {}
for fname, key in _preproc_files.items():
    fpath = os.path.join(PREPROC_DIR, fname)
    if not os.path.exists(fpath):
        msg = f"Missing: {fpath}"
        LOAD_ERRORS.append(msg)
        print(f"  ❌ {msg}")
    else:
        try:
            with open(fpath, "rb") as f:
                _loaded[key] = pickle.load(f)
            print(f"  ✅ {fname}")
        except Exception as e:
            msg = f"Load error {fname}: {e}"
            LOAD_ERRORS.append(msg)
            print(f"  ❌ {msg}")

if "scaler"              in _loaded: scaler              = _loaded["scaler"]
if "label_encoders"      in _loaded: label_encoders      = _loaded["label_encoders"]
if "static_feature_cols" in _loaded: static_feature_cols = _loaded["static_feature_cols"]
if "seq_params"          in _loaded:
    MAX_SEQ_LEN = _loaded["seq_params"]["max_seq_len"]
    n_features  = _loaded["seq_params"]["n_features"]

preprocessors_ok = all(
    v is not None for v in [scaler, label_encoders, static_feature_cols, MAX_SEQ_LEN, n_features]
)
print("✅ Preprocessors ready." if preprocessors_ok else "⚠️  Preprocessors incomplete — /predict may fail.")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Load models
# ─────────────────────────────────────────────────────────────────────────────
MODELS = {}
MODEL_NAMES = ["ANN", "LSTM", "Bi-LSTM", "GRU", "CNN-LSTM"]

print("Loading models…")
# Import TF once, quietly
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    TF_OK = True
except ImportError as e:
    LOAD_ERRORS.append(f"TensorFlow import failed: {e}")
    print(f"  ❌ TensorFlow not available: {e}")
    TF_OK = False

if TF_OK:
    for name in MODEL_NAMES:
        path = os.path.join(MODEL_DIR, f"{name}.keras")
        if not os.path.exists(path):
            print(f"  ⚠️  {name}.keras not found at {path}")
        else:
            try:
                MODELS[name] = tf.keras.models.load_model(path)
                print(f"  ✅ {name}")
            except Exception as e:
                msg = f"Model load error {name}: {e}"
                LOAD_ERRORS.append(msg)
                print(f"  ❌ {msg}")

print(f"Models loaded: {list(MODELS.keys()) or 'none'}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Preprocessing helper
# ─────────────────────────────────────────────────────────────────────────────
_CAT_COLS = ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']
_NUM_COLS = ['num_of_prev_attempts', 'studied_credits', 'avg_score',
             'submission_count', 'total_clicks', 'active_days', 'engagement_ratio']

def preprocess_static(data: dict) -> np.ndarray:
    """
    Convert raw JSON dict to a scaled static feature array (1, n_features).
    Raises RuntimeError if preprocessors are not loaded.
    """
    if not preprocessors_ok:
        raise RuntimeError(
            "Preprocessors not loaded. Ensure the preprocessors/ directory exists "
            "next to app.py and contains the required .pkl files."
        )

    row = {}
    for col in _CAT_COLS:
        val = str(data.get(col, ""))
        le  = label_encoders.get(col)
        if le is not None:
            known = set(le.classes_)
            val   = val if val in known else le.classes_[0]
            row[col] = int(le.transform([val])[0])
        else:
            row[col] = 0

    for col in _NUM_COLS:
        row[col] = float(data.get(col, 0))

    df_row = pd.DataFrame([row])[static_feature_cols]
    df_row = df_row.copy()
    df_row[_NUM_COLS] = scaler.transform(df_row[_NUM_COLS])
    return df_row.values.astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# 4. /predict  (POST)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body (JSON):
      {
        "model": "ANN" | "LSTM" | "Bi-LSTM" | "GRU" | "CNN-LSTM",
        "gender": "M",
        "region": "South East Region",
        "highest_education": "A Level or Equivalent",
        "imd_band": "50-60%",
        "age_band": "35-55",
        "disability": "N",
        "num_of_prev_attempts": 0,
        "studied_credits": 60,
        "avg_score": 72.5,
        "submission_count": 5,
        "total_clicks": 3400,
        "active_days": 120,
        "engagement_ratio": 28.3
      }
    Returns:
      { "model": str, "prediction": 0|1, "probability": float, "label": str }
    """
    try:
        data       = request.get_json(force=True, silent=True) or {}
        model_name = str(data.get("model", "ANN")).upper()

        if not MODELS:
            return jsonify({
                "error": "No models are loaded. Ensure models/ directory exists with .keras files.",
                "load_errors": LOAD_ERRORS,
            }), 503

        if model_name not in MODELS:
            return jsonify({
                "error": f"Model '{model_name}' not available.",
                "available": list(MODELS.keys()),
            }), 400

        mdl = MODELS[model_name]
        X_static = preprocess_static(data)          # (1, n_static_features)

        if model_name == "ANN":
            X = X_static
        else:
            # Sequence models: broadcast static features over time dimension
            X = np.zeros((1, MAX_SEQ_LEN, n_features), dtype=np.float32)
            F = min(X_static.shape[1], n_features)
            X[0, :, :F] = X_static[0, :F]

        prob = float(mdl.predict(X, verbose=0)[0][0])
        pred = int(prob > 0.5)

        return jsonify({
            "model":       model_name,
            "prediction":  pred,
            "probability": round(prob, 4),
            "label":       "Pass/Distinction" if pred == 1 else "Fail/Withdrawn",
        })

    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─────────────────────────────────────────────────────────────────────────────
# 5. /shap  (GET)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/shap", methods=["GET"])
def shap_endpoint():
    """
    GET /shap
    Returns pre-computed SHAP results (top features + image paths).
    Run shap_analysis.py first to generate these files.
    """
    fi_path      = os.path.join(RESULTS_DIR, "gru_feature_importance.csv")
    summary_img  = os.path.join(PLOT_DIR,    "gru_shap_summary.png")
    bar_img      = os.path.join(PLOT_DIR,    "gru_shap_bar.png")
    shap_npy     = os.path.join(RESULTS_DIR, "gru_shap_values.npy")

    errors        = []
    top_features  = []

    # Load feature importance CSV
    if os.path.exists(fi_path):
        try:
            fi_df = pd.read_csv(fi_path)
            top_features = [
                {
                    "rank":          int(i + 1),
                    "feature":       str(row["feature"]),
                    "mean_abs_shap": round(float(row["mean_abs_shap"]), 6),
                }
                for i, row in fi_df.head(10).iterrows()
            ]
        except Exception as e:
            errors.append(f"Could not read {fi_path}: {e}")
    else:
        errors.append(f"Feature importance CSV not found: {fi_path} — run shap_analysis.py first.")

    # Verify assets
    for label, path in [("summary_plot", summary_img), ("bar_plot", bar_img),
                         ("shap_values_npy", shap_npy)]:
        if not os.path.exists(path):
            errors.append(f"{label} not found: {path}")

    return jsonify({
        "model":          "GRU",
        "explainer_note": "Computed by shap_analysis.py (DeepExplainer → GradientExplainer → KernelExplainer fallback)",
        "top_features":   top_features,
        "files": {
            "summary_plot":    summary_img,
            "bar_plot":        bar_img,
            "importance_csv":  fi_path,
            "shap_values_npy": shap_npy,
        },
        "errors": errors,
        "ready":  len(errors) == 0,
    })

# ─────────────────────────────────────────────────────────────────────────────
# 6. /fairness  (GET)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/fairness", methods=["GET"])
def fairness_endpoint():
    """
    GET /fairness
    Returns pre-computed fairness metrics and chart paths.
    Run fairness_analysis.py first to generate these files.
    """
    ATTRS = ["gender", "region", "disability"]

    errors          = []
    fairness_metrics = {}
    group_details   = {}
    charts          = {}

    # Summary CSV
    summary_path = os.path.join(RESULTS_DIR, "fairness_summary.csv")
    if os.path.exists(summary_path):
        try:
            sdf = pd.read_csv(summary_path)
            for _, row in sdf.iterrows():
                attr = str(row.get("sensitive_attribute", "unknown"))
                di   = row.get("disparate_impact_ratio", float("nan"))
                fairness_metrics[attr] = {
                    "demographic_parity_difference": float(row.get("demographic_parity_difference", float("nan"))),
                    "equal_opportunity_difference":  float(row.get("equal_opportunity_difference",  float("nan"))),
                    "disparate_impact_ratio":        float(di),
                    "n_groups":                      int(row.get("n_groups", 0)),
                    "adverse_impact_flag":           (float(di) < 0.8) if not pd.isna(di) else False,
                }
        except Exception as e:
            errors.append(f"Could not read fairness_summary.csv: {e}")
    else:
        errors.append(f"fairness_summary.csv not found: {summary_path} — run fairness_analysis.py first.")

    # Per-attribute details
    for attr in ATTRS:
        chart = os.path.join(PLOT_DIR,    f"fairness_{attr}.png")
        csv   = os.path.join(RESULTS_DIR, f"fairness_{attr}.csv")
        charts[attr] = chart

        if not os.path.exists(chart):
            errors.append(f"Chart not found: {chart}")

        if os.path.exists(csv):
            try:
                gdf = pd.read_csv(csv)
                group_details[attr] = gdf.to_dict(orient="records")
            except Exception as e:
                errors.append(f"Could not read {csv}: {e}")
        else:
            errors.append(f"Group CSV not found: {csv}")

    return jsonify({
        "model":           "GRU",
        "fairness_metrics": fairness_metrics,
        "group_details":    group_details,
        "charts":           charts,
        "errors":           errors,
        "ready":            len(errors) == 0,
    })

# ─────────────────────────────────────────────────────────────────────────────
# 7. /health  (GET)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":            "ok",
        "base_dir":          BASE_DIR,
        "models_loaded":     list(MODELS.keys()),
        "preprocessors_ok":  preprocessors_ok,
        "shap_ready":        os.path.exists(os.path.join(RESULTS_DIR, "gru_feature_importance.csv")),
        "fairness_ready":    os.path.exists(os.path.join(RESULTS_DIR, "fairness_summary.csv")),
        "startup_errors":    LOAD_ERRORS,
    })

# ─────────────────────────────────────────────────────────────────────────────
# 8. /models  (GET) – convenience list
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/models", methods=["GET"])
def list_models():
    return jsonify({
        "available": list(MODELS.keys()),
        "expected":  MODEL_NAMES,
    })

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 Flask app starting on http://0.0.0.0:{port}")
    print(f"   BASE_DIR : {BASE_DIR}")
    print(f"   Endpoints: GET /health  GET /models  GET /shap  GET /fairness")
    print(f"              POST /predict")
    app.run(host="0.0.0.0", port=port, debug=False)