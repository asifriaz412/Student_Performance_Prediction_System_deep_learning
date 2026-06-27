"""
SHAP Explainability Analysis for Student Performance Prediction
Analyses the best model (GRU) using SHAP values.
Falls back: DeepExplainer -> GradientExplainer -> KernelExplainer
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Directories ──────────────────────────────────────────────────────────────
MODEL_DIR   = "models"
PREPROC_DIR = "preprocessors"
PLOT_DIR    = "plots"
RESULTS_DIR = "results"
DATA_PATH   = "data/"

os.makedirs(PLOT_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Feature names (must match training order) ─────────────────────────────────
FEATURE_NAMES = [
    'gender', 'region', 'highest_education', 'imd_band',
    'age_band', 'disability', 'num_of_prev_attempts', 'studied_credits',
    'avg_score', 'submission_count', 'total_clicks', 'active_days',
    'engagement_ratio'
]

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load preprocessors
# ─────────────────────────────────────────────────────────────────────────────
print("Loading preprocessors...")
try:
    with open(os.path.join(PREPROC_DIR, "static_scaler.pkl"),     "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(PREPROC_DIR, "label_encoders.pkl"),    "rb") as f:
        label_encoders = pickle.load(f)
    with open(os.path.join(PREPROC_DIR, "static_feature_cols.pkl"), "rb") as f:
        static_feature_cols = pickle.load(f)
    with open(os.path.join(PREPROC_DIR, "seq_params.pkl"),        "rb") as f:
        seq_params = pickle.load(f)
    MAX_SEQ_LEN = seq_params['max_seq_len']
    n_features  = seq_params['n_features']
    print("✅ Preprocessors loaded.")
except Exception as e:
    print(f"❌ Failed to load preprocessors: {e}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Load GRU model
# ─────────────────────────────────────────────────────────────────────────────
print("Loading GRU model...")
try:
    import tensorflow as tf
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "GRU.keras"))
    print(f"✅ GRU model loaded  (input shape: {model.input_shape})")
except Exception as e:
    print(f"❌ Failed to load GRU model: {e}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Rebuild test sequences from raw data (same pipeline as training)
# ─────────────────────────────────────────────────────────────────────────────
print("Rebuilding test data from raw OULAD files...")
try:
    info        = pd.read_csv(DATA_PATH + "studentInfo.csv")
    assessment  = pd.read_csv(DATA_PATH + "studentAssessment.csv")
    vle         = pd.read_csv(DATA_PATH + "studentVle.csv")
    assessments = pd.read_csv(DATA_PATH + "assessments.csv")

    assessment['score']   = pd.to_numeric(assessment['score'],   errors='coerce')
    vle['sum_click']      = pd.to_numeric(vle['sum_click'],      errors='coerce')

    # ── Static features ───────────────────────────────────────────────────────
    assess_with_course = assessment.merge(
        assessments[['id_assessment', 'code_module', 'code_presentation']],
        on='id_assessment', how='left'
    )
    assess_agg = assess_with_course.groupby(
        ['id_student', 'code_module', 'code_presentation']
    ).agg(avg_score=('score', 'mean'), submission_count=('id_assessment', 'count')).reset_index()

    vle_agg = vle.groupby(
        ['id_student', 'code_module', 'code_presentation']
    ).agg(total_clicks=('sum_click', 'sum'), active_days=('date', 'nunique')).reset_index()
    vle_agg['engagement_ratio'] = vle_agg['total_clicks'] / vle_agg['active_days'].replace(0, np.nan)

    df = info.merge(assess_agg, on=['id_student', 'code_module', 'code_presentation'], how='left')
    df = df.merge(vle_agg,     on=['id_student', 'code_module', 'code_presentation'], how='left')

    for col in ['avg_score', 'submission_count', 'total_clicks', 'active_days', 'engagement_ratio']:
        df[col] = df[col].fillna(0)

    cat_cols = ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        le = label_encoders.get(col)
        if le:
            # handle unseen labels gracefully
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(df[col])
        else:
            from sklearn.preprocessing import LabelEncoder
            tmp_le = LabelEncoder()
            df[col] = tmp_le.fit_transform(df[col].astype(str))

    TARGET = "final_result"
    df[TARGET] = df[TARGET].map({'Pass': 1, 'Distinction': 1, 'Fail': 0, 'Withdrawn': 0})
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    X_static_full = df[static_feature_cols].copy()
    num_cols = ['num_of_prev_attempts', 'studied_credits', 'avg_score',
                'submission_count', 'total_clicks', 'active_days', 'engagement_ratio']
    X_static_full[num_cols] = scaler.transform(X_static_full[num_cols])
    y_full = df[TARGET].values

    # ── Build sequences ───────────────────────────────────────────────────────
    vle_daily = vle.groupby(
        ['id_student', 'code_module', 'code_presentation', 'date']
    ).agg(sum_click=('sum_click', 'sum'), activity_count=('id_site', 'nunique')
    ).reset_index().rename(columns={'date': 'day'})

    assess_dates = assessment.merge(
        assessments[['id_assessment', 'code_module', 'code_presentation', 'date']],
        on='id_assessment', how='left'
    )
    assess_dates = assess_dates[['id_student', 'code_module', 'code_presentation',
                                 'date', 'score']].rename(columns={'date': 'day'})

    demo_cols = ['gender', 'region', 'highest_education', 'imd_band',
                 'age_band', 'disability', 'num_of_prev_attempts', 'studied_credits']

    with open(os.path.join(PREPROC_DIR, "demo_scaler.pkl"), "rb") as f:
        demo_scaler = pickle.load(f)

    df_enc = df.copy()
    df_enc[['num_of_prev_attempts', 'studied_credits']] = demo_scaler.transform(
        df_enc[['num_of_prev_attempts', 'studied_credits']]
    )
    demo_dict = {}
    for _, row in df_enc.iterrows():
        key = (row['id_student'], row['code_module'], row['code_presentation'])
        demo_dict[key] = row[demo_cols].values.astype(np.float32)

    seq_list, y_seq_list, instance_keys = [], [], []
    for key, vg in vle_daily.groupby(['id_student', 'code_module', 'code_presentation']):
        demo_vec = demo_dict.get(key)
        if demo_vec is None:
            continue
        ai = assess_dates[
            (assess_dates['id_student'] == key[0]) &
            (assess_dates['code_module'] == key[1]) &
            (assess_dates['code_presentation'] == key[2])
        ]
        score_dict = dict(zip(ai['day'], ai['score']))
        all_days = vg['day'].values
        if len(all_days) == 0:
            day_range = np.array([0])
        else:
            day_range = np.arange(all_days.min(), all_days.max() + 1)

        sc_arr = np.zeros(len(day_range)); ac_arr = np.zeros(len(day_range))
        sco_arr = np.zeros(len(day_range)); last = 0.0
        vdm = vg.set_index('day')
        for i, day in enumerate(day_range):
            if day in vdm.index:
                sc_arr[i] = vdm.loc[day, 'sum_click']
                ac_arr[i] = vdm.loc[day, 'activity_count']
            if day in score_dict: last = score_dict[day]
            sco_arr[i] = last

        tv = np.stack([sc_arr, ac_arr, sco_arr], axis=1)
        dr = np.tile(demo_vec, (len(day_range), 1))
        seq_list.append(np.concatenate([tv, dr], axis=1))

        label_rows = df_enc[
            (df_enc['id_student'] == key[0]) &
            (df_enc['code_module'] == key[1]) &
            (df_enc['code_presentation'] == key[2])
        ]
        if label_rows.empty:
            continue
        y_seq_list.append(label_rows[TARGET].values[0])
        instance_keys.append(key)

    # Pad to MAX_SEQ_LEN
    X_seq = np.zeros((len(seq_list), MAX_SEQ_LEN, n_features))
    for i, s in enumerate(seq_list):
        X_seq[i, :s.shape[0], :] = s
    y_seq = np.array(y_seq_list, dtype=np.int32)

    # Match static rows that have sequences
    keys_order = list(zip(df['id_student'], df['code_module'], df['code_presentation']))
    seq_key_idx = {k: i for i, k in enumerate(instance_keys)}
    has_seq = [k in seq_key_idx for k in keys_order]
    y_seq_full = y_full[has_seq]
    seq_idx_arr = np.array([seq_key_idx[keys_order[i]] for i in range(len(keys_order)) if has_seq[i]])

    from sklearn.model_selection import train_test_split
    _, test_seq_idx, _, y_test_seq = train_test_split(
        np.arange(len(y_seq_full)), y_seq_full,
        test_size=0.2, stratify=y_seq_full, random_state=42
    )
    X_test_seq = X_seq[seq_idx_arr[test_seq_idx]]
    print(f"✅ Test sequences built: {X_test_seq.shape}")

except Exception as e:
    print(f"❌ Data rebuild failed: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Time-step feature names  (3 temporal + 8 demo repeated)
# ─────────────────────────────────────────────────────────────────────────────
temporal_names = ['sum_click', 'activity_count', 'running_score']
demo_names     = ['gender', 'region', 'highest_education', 'imd_band',
                  'age_band', 'disability', 'num_of_prev_attempts', 'studied_credits']
seq_feature_names = temporal_names + demo_names   # 11 features per timestep

# For SHAP we flatten or use mean-over-timesteps
def get_mean_features(X_3d):
    """Return mean over time axis → (N, F)"""
    return X_3d.mean(axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# 5. SHAP Analysis with fallback chain
# ─────────────────────────────────────────────────────────────────────────────
print("\nRunning SHAP analysis...")
try:
    import shap
except ImportError:
    print("Installing shap...")
    os.system(f"{sys.executable} -m pip install shap --quiet")
    import shap

shap_values = None
explainer_used = None
N_BACKGROUND = min(100, X_test_seq.shape[0])
N_EXPLAIN    = min(200, X_test_seq.shape[0])

background = X_test_seq[:N_BACKGROUND]
explain_data = X_test_seq[:N_EXPLAIN]

# ── Attempt 1: DeepExplainer ─────────────────────────────────────────────────
try:
    print("Trying shap.DeepExplainer...")
    explainer = shap.DeepExplainer(model, background)
    raw = explainer.shap_values(explain_data)
    # raw may be list[array] or array
    if isinstance(raw, list):
        raw = raw[0]
    # shape: (N, T, F) or (N, F)
    if raw.ndim == 3:
        shap_values = raw.mean(axis=1)   # average over timesteps → (N, F)
    else:
        shap_values = raw
    explainer_used = "DeepExplainer"
    print("✅ DeepExplainer succeeded.")
except Exception as e1:
    print(f"  DeepExplainer failed: {e1}")

    # ── Attempt 2: GradientExplainer ─────────────────────────────────────────
    try:
        print("Trying shap.GradientExplainer...")
        explainer = shap.GradientExplainer(model, background)
        raw = explainer.shap_values(explain_data)
        if isinstance(raw, list):
            raw = raw[0]
        shap_values = raw.mean(axis=1) if raw.ndim == 3 else raw
        explainer_used = "GradientExplainer"
        print("✅ GradientExplainer succeeded.")
    except Exception as e2:
        print(f"  GradientExplainer failed: {e2}")

        # ── Attempt 3: KernelExplainer (on mean-flattened data) ───────────────
        try:
            print("Trying shap.KernelExplainer (flattened mean features)...")
            X_bg_flat  = get_mean_features(background)
            X_exp_flat = get_mean_features(explain_data[:50])   # keep small for speed

            def predict_flat(X_flat):
                # Broadcast mean features back to (N, T, F)  — constant in time
                X_3d = np.tile(X_flat[:, np.newaxis, :], (1, MAX_SEQ_LEN, 1))
                return model.predict(X_3d, verbose=0).flatten()

            explainer = shap.KernelExplainer(predict_flat, X_bg_flat)
            shap_values = explainer.shap_values(X_exp_flat, nsamples=50, silent=True)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            explainer_used = "KernelExplainer"
            explain_data = explain_data[:50]   # match size
            print("✅ KernelExplainer succeeded.")
        except Exception as e3:
            print(f"❌ All SHAP methods failed: {e3}")
            sys.exit(1)

print(f"Explainer used: {explainer_used}  |  SHAP shape: {shap_values.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Feature importance from SHAP values
# ─────────────────────────────────────────────────────────────────────────────
F = shap_values.shape[1]
names = (seq_feature_names + [f"feat_{i}" for i in range(F - len(seq_feature_names))])[:F]

mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({'feature': names, 'mean_abs_shap': mean_abs_shap})
importance_df = importance_df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

top20 = importance_df.head(20)
top20.to_csv(os.path.join(RESULTS_DIR, "gru_feature_importance.csv"), index=False)
np.save(os.path.join(RESULTS_DIR, "gru_shap_values.npy"), shap_values)
print(f"✅ SHAP values saved to results/gru_shap_values.npy")

print("\n📊 Top 10 Most Important Features:")
print(importance_df.head(10).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 7. SHAP Summary Plot
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating SHAP Summary Plot...")
try:
    # Use mean features for shap.summary_plot
    if explainer_used == "KernelExplainer":
        X_flat_for_plot = get_mean_features(explain_data)
    else:
        X_flat_for_plot = get_mean_features(explain_data)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_flat_for_plot,
                      feature_names=names, show=False, max_display=15)
    plt.title("SHAP Summary Plot – GRU Model", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "gru_shap_summary.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Saved plots/gru_shap_summary.png")
except Exception as e:
    print(f"  Summary plot error: {e}  — using manual fallback.")
    plt.figure(figsize=(10, 8))
    plt.barh(names[:15][::-1], mean_abs_shap[:15][::-1], color='steelblue')
    plt.xlabel("Mean |SHAP value|")
    plt.title("SHAP Feature Importance – GRU Model")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "gru_shap_summary.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Saved fallback summary plot.")

# ─────────────────────────────────────────────────────────────────────────────
# 8. SHAP Bar Plot (top 20)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating SHAP Bar Plot...")
try:
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, get_mean_features(explain_data),
                      feature_names=names, plot_type='bar',
                      show=False, max_display=20)
    plt.title("SHAP Bar Plot – GRU Model (Top 20 Features)", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "gru_shap_bar.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Saved plots/gru_shap_bar.png")
except Exception as e:
    print(f"  Bar plot error: {e}  — using manual fallback.")
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top20)))[::-1]
    ax.barh(top20['feature'][::-1], top20['mean_abs_shap'][::-1], color=colors)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("SHAP Feature Importance – GRU Model (Top 20)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "gru_shap_bar.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Saved fallback bar plot.")

print("\n✅ SHAP analysis complete.")
print(f"   Explainer : {explainer_used}")
print(f"   Plots     : plots/gru_shap_summary.png, plots/gru_shap_bar.png")
print(f"   CSV       : results/gru_feature_importance.csv")
print(f"   NPY       : results/gru_shap_values.npy")