"""
Fairness Evaluation for Student Performance Prediction
Evaluates the best model (GRU) across sensitive attributes:
  gender, region, disability
Metrics: group-wise Accuracy/Precision/Recall/F1
         + Demographic Parity Difference
         + Equal Opportunity Difference
         + Disparate Impact Ratio
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

# ── Directories ───────────────────────────────────────────────────────────────
MODEL_DIR   = "models"
PREPROC_DIR = "preprocessors"
PLOT_DIR    = "plots"
RESULTS_DIR = "results"
DATA_PATH   = "data/"

os.makedirs(PLOT_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load preprocessors
# ─────────────────────────────────────────────────────────────────────────────
print("Loading preprocessors...")
try:
    with open(os.path.join(PREPROC_DIR, "static_scaler.pkl"),      "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(PREPROC_DIR, "label_encoders.pkl"),     "rb") as f:
        label_encoders = pickle.load(f)
    with open(os.path.join(PREPROC_DIR, "static_feature_cols.pkl"),"rb") as f:
        static_feature_cols = pickle.load(f)
    with open(os.path.join(PREPROC_DIR, "seq_params.pkl"),         "rb") as f:
        seq_params = pickle.load(f)
    with open(os.path.join(PREPROC_DIR, "demo_scaler.pkl"),        "rb") as f:
        demo_scaler = pickle.load(f)
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
    print("✅ GRU model loaded.")
except Exception as e:
    print(f"❌ Failed to load GRU model: {e}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Rebuild test data
# ─────────────────────────────────────────────────────────────────────────────
print("Rebuilding test data...")
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    info        = pd.read_csv(DATA_PATH + "studentInfo.csv")
    assessment  = pd.read_csv(DATA_PATH + "studentAssessment.csv")
    vle         = pd.read_csv(DATA_PATH + "studentVle.csv")
    assessments = pd.read_csv(DATA_PATH + "assessments.csv")

    assessment['score'] = pd.to_numeric(assessment['score'], errors='coerce')
    vle['sum_click']    = pd.to_numeric(vle['sum_click'],    errors='coerce')

    # --- static aggregation ---
    awc = assessment.merge(
        assessments[['id_assessment','code_module','code_presentation']],
        on='id_assessment', how='left'
    )
    assess_agg = awc.groupby(['id_student','code_module','code_presentation']).agg(
        avg_score=('score','mean'), submission_count=('id_assessment','count')
    ).reset_index()

    vle_agg = vle.groupby(['id_student','code_module','code_presentation']).agg(
        total_clicks=('sum_click','sum'), active_days=('date','nunique')
    ).reset_index()
    vle_agg['engagement_ratio'] = vle_agg['total_clicks'] / vle_agg['active_days'].replace(0, np.nan)

    df = info.merge(assess_agg, on=['id_student','code_module','code_presentation'], how='left')
    df = df.merge(vle_agg,     on=['id_student','code_module','code_presentation'], how='left')
    for c in ['avg_score','submission_count','total_clicks','active_days','engagement_ratio']:
        df[c] = df[c].fillna(0)

    # Keep raw sensitive attributes BEFORE encoding
    df['gender_raw']     = df['gender'].fillna(df['gender'].mode()[0])
    df['region_raw']     = df['region'].fillna(df['region'].mode()[0])
    df['disability_raw'] = df['disability'].fillna(df['disability'].mode()[0])

    cat_cols = ['gender','region','highest_education','imd_band','age_band','disability']
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        le = label_encoders.get(col)
        if le:
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(df[col])
        else:
            tmp = LabelEncoder(); df[col] = tmp.fit_transform(df[col].astype(str))

    TARGET = "final_result"
    df[TARGET] = df[TARGET].map({'Pass':1,'Distinction':1,'Fail':0,'Withdrawn':0})
    df = df.dropna(subset=[TARGET]); df[TARGET] = df[TARGET].astype(int)

    X_static_full = df[static_feature_cols].copy()
    num_cols = ['num_of_prev_attempts','studied_credits','avg_score',
                'submission_count','total_clicks','active_days','engagement_ratio']
    X_static_full[num_cols] = scaler.transform(X_static_full[num_cols])
    y_full = df[TARGET].values

    # --- sequence creation ---
    vle_daily = vle.groupby(['id_student','code_module','code_presentation','date']).agg(
        sum_click=('sum_click','sum'), activity_count=('id_site','nunique')
    ).reset_index().rename(columns={'date':'day'})

    assess_dates = assessment.merge(
        assessments[['id_assessment','code_module','code_presentation','date']],
        on='id_assessment', how='left'
    )
    assess_dates = assess_dates[['id_student','code_module','code_presentation',
                                 'date','score']].rename(columns={'date':'day'})

    demo_cols = ['gender','region','highest_education','imd_band',
                 'age_band','disability','num_of_prev_attempts','studied_credits']
    df_enc = df.copy()
    df_enc[['num_of_prev_attempts','studied_credits']] = demo_scaler.transform(
        df_enc[['num_of_prev_attempts','studied_credits']]
    )
    demo_dict = {}
    for _, row in df_enc.iterrows():
        key = (row['id_student'], row['code_module'], row['code_presentation'])
        demo_dict[key] = row[demo_cols].values.astype(np.float32)

    seq_list, y_seq_list, instance_keys = [], [], []
    for key, vg in vle_daily.groupby(['id_student','code_module','code_presentation']):
        dv = demo_dict.get(key)
        if dv is None: continue
        ai = assess_dates[
            (assess_dates['id_student']==key[0]) &
            (assess_dates['code_module']==key[1]) &
            (assess_dates['code_presentation']==key[2])
        ]
        sd = dict(zip(ai['day'], ai['score']))
        all_days = vg['day'].values
        dr = np.array([0]) if len(all_days)==0 else np.arange(all_days.min(), all_days.max()+1)
        sc=np.zeros(len(dr)); ac=np.zeros(len(dr)); sco=np.zeros(len(dr)); last=0.
        vdm = vg.set_index('day')
        for i, day in enumerate(dr):
            if day in vdm.index:
                sc[i]=vdm.loc[day,'sum_click']; ac[i]=vdm.loc[day,'activity_count']
            if day in sd: last=sd[day]
            sco[i]=last
        tv = np.stack([sc,ac,sco],axis=1)
        dri = np.tile(dv,(len(dr),1))
        seq_list.append(np.concatenate([tv,dri],axis=1))
        lr = df_enc[
            (df_enc['id_student']==key[0])&(df_enc['code_module']==key[1])&
            (df_enc['code_presentation']==key[2])
        ]
        if lr.empty: continue
        y_seq_list.append(lr[TARGET].values[0])
        instance_keys.append(key)

    X_seq = np.zeros((len(seq_list), MAX_SEQ_LEN, n_features))
    for i, s in enumerate(seq_list):
        X_seq[i, :s.shape[0], :] = s
    y_seq = np.array(y_seq_list, dtype=np.int32)

    keys_order = list(zip(df['id_student'],df['code_module'],df['code_presentation']))
    seq_key_idx = {k:i for i,k in enumerate(instance_keys)}
    has_seq = [k in seq_key_idx for k in keys_order]
    y_seq_full   = y_full[has_seq]
    seq_idx_arr  = np.array([seq_key_idx[keys_order[i]] for i in range(len(keys_order)) if has_seq[i]])

    # Sensitive features for the same subset
    df_has_seq = df[has_seq].reset_index(drop=True)

    _, test_seq_idx, _, y_test_seq = train_test_split(
        np.arange(len(y_seq_full)), y_seq_full,
        test_size=0.2, stratify=y_seq_full, random_state=42
    )
    X_test_seq = X_seq[seq_idx_arr[test_seq_idx]]
    df_test     = df_has_seq.iloc[test_seq_idx].reset_index(drop=True)
    y_test      = y_test_seq

    print(f"✅ Test data ready: {X_test_seq.shape[0]} samples")

except Exception as e:
    print(f"❌ Data rebuild failed: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Predict
# ─────────────────────────────────────────────────────────────────────────────
print("Generating predictions...")
y_pred_proba = model.predict(X_test_seq, verbose=0).flatten()
y_pred       = (y_pred_proba > 0.5).astype(int)
print(f"✅ Predictions done. Overall accuracy: {(y_pred==y_test).mean():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Fairness evaluation function
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_fairness(y_true, y_pred, sensitive_feature, feature_name="feature"):
    """
    Compute group-wise metrics and formal fairness measures.

    Parameters
    ----------
    y_true            : array-like of true labels (0/1)
    y_pred            : array-like of predicted labels (0/1)
    sensitive_feature : array-like of group membership
    feature_name      : human-readable name for console output

    Returns
    -------
    group_df  : DataFrame with per-group Accuracy/Precision/Recall/F1
    summary   : dict with formal fairness metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sf     = np.array(sensitive_feature)

    groups = np.unique(sf)
    rows   = []

    for g in groups:
        mask = sf == g
        yt, yp = y_true[mask], y_pred[mask]
        n = mask.sum()
        if n == 0:
            continue
        acc  = accuracy_score(yt, yp)
        prec = precision_score(yt, yp, zero_division=0)
        rec  = recall_score(yt, yp, zero_division=0)
        f1   = f1_score(yt, yp, zero_division=0)
        pos_rate     = yp.mean()           # positive prediction rate
        tpr          = rec                  # True Positive Rate (for class 1)
        rows.append({
            'group': g,
            'n_samples': int(n),
            'accuracy':  round(acc,  4),
            'precision': round(prec, 4),
            'recall':    round(rec,  4),
            'f1_score':  round(f1,   4),
            'positive_rate': round(pos_rate, 4),
            'tpr':           round(tpr, 4),
        })

    group_df = pd.DataFrame(rows)

    # ── Formal fairness metrics ───────────────────────────────────────────────
    # We pick the group with the HIGHEST positive_rate as the reference group
    # (or the majority group if preferred — here highest for conservative measure)
    pos_rates = group_df['positive_rate'].values
    tprs      = group_df['tpr'].values

    # Demographic Parity Difference: max - min of positive prediction rates
    dp_diff = float(pos_rates.max() - pos_rates.min())

    # Equal Opportunity Difference: max - min of TPR (recall) across groups
    eo_diff = float(tprs.max() - tprs.min())

    # Disparate Impact Ratio: min_rate / max_rate  (1.0 is perfect; <0.8 is concerning)
    max_pos = pos_rates.max()
    di_ratio = float(pos_rates.min() / max_pos) if max_pos > 0 else float('nan')

    summary = {
        'sensitive_attribute':          feature_name,
        'demographic_parity_difference': round(dp_diff,  4),
        'equal_opportunity_difference':  round(eo_diff,  4),
        'disparate_impact_ratio':        round(di_ratio, 4),
        'n_groups': len(group_df),
    }

    print(f"\n── Fairness: {feature_name} ──────────────────────────")
    print(group_df.to_string(index=False))
    print(f"  Demographic Parity Difference : {dp_diff:.4f}")
    print(f"  Equal Opportunity Difference  : {eo_diff:.4f}")
    print(f"  Disparate Impact Ratio        : {di_ratio:.4f}")
    if di_ratio < 0.8:
        print(f"  ⚠️  Disparate Impact < 0.8 — potential adverse impact detected.")

    return group_df, summary

# ─────────────────────────────────────────────────────────────────────────────
# 6. Run fairness evaluations
# ─────────────────────────────────────────────────────────────────────────────
sensitive_attrs = {
    'gender':     df_test['gender_raw'].values,
    'region':     df_test['region_raw'].values,
    'disability': df_test['disability_raw'].values,
}

all_summaries = []

for attr_name, sf_values in sensitive_attrs.items():
    print(f"\nEvaluating fairness for: {attr_name}")
    try:
        group_df, summary = evaluate_fairness(
            y_test, y_pred, sf_values, feature_name=attr_name
        )
        csv_path = os.path.join(RESULTS_DIR, f"fairness_{attr_name}.csv")
        group_df.to_csv(csv_path, index=False)
        print(f"✅ Saved {csv_path}")
        all_summaries.append(summary)
    except Exception as e:
        print(f"❌ Fairness eval failed for {attr_name}: {e}")

# Save summary
summary_df = pd.DataFrame(all_summaries)
summary_df.to_csv(os.path.join(RESULTS_DIR, "fairness_summary.csv"), index=False)
print(f"\n✅ Saved results/fairness_summary.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Visualisations (matplotlib only)
# ─────────────────────────────────────────────────────────────────────────────
METRIC_COLS = ['accuracy', 'precision', 'recall', 'f1_score']
COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

def plot_fairness(attr_name):
    csv_path = os.path.join(RESULTS_DIR, f"fairness_{attr_name}.csv")
    if not os.path.exists(csv_path):
        print(f"  Skipping plot — {csv_path} not found.")
        return

    gdf = pd.read_csv(csv_path)
    groups = [str(g) for g in gdf['group'].tolist()]
    n_groups  = len(groups)
    n_metrics = len(METRIC_COLS)
    bar_width = 0.18
    x = np.arange(n_groups)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Fairness Analysis – {attr_name.capitalize()}", fontsize=14, fontweight='bold')

    # ── Left: grouped bar chart of metrics ────────────────────────────────────
    ax = axes[0]
    for i, (metric, color) in enumerate(zip(METRIC_COLS, COLORS)):
        vals = gdf[metric].values
        ax.bar(x + i * bar_width, vals, bar_width, label=metric.capitalize(), color=color, alpha=0.85)
    ax.set_xlabel(attr_name.capitalize(), fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Group-wise Performance Metrics")
    ax.set_xticks(x + bar_width * (n_metrics - 1) / 2)
    ax.set_xticklabels(groups, rotation=25, ha='right', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # ── Right: positive rate (Demographic Parity) ─────────────────────────────
    ax2 = axes[1]
    colors_pr = plt.cm.coolwarm(np.linspace(0.2, 0.8, n_groups))
    ax2.bar(groups, gdf['positive_rate'].values, color=colors_pr, alpha=0.85)
    ax2.axhline(gdf['positive_rate'].mean(), color='black', linestyle='--',
                linewidth=1.2, label=f"Mean = {gdf['positive_rate'].mean():.3f}")
    ax2.set_xlabel(attr_name.capitalize(), fontsize=11)
    ax2.set_ylabel("Positive Prediction Rate", fontsize=11)
    ax2.set_title("Demographic Parity (Positive Rate by Group)")
    ax2.set_xticklabels(groups, rotation=25, ha='right', fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f"fairness_{attr_name}.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved {out}")

for attr in ['gender', 'region', 'disability']:
    try:
        plot_fairness(attr)
    except Exception as e:
        print(f"❌ Plot failed for {attr}: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Done
# ─────────────────────────────────────────────────────────────────────────────
print("\n✅ Fairness analysis complete.")
print("   CSVs  : results/fairness_gender.csv, fairness_region.csv, fairness_disability.csv")
print("   CSV   : results/fairness_summary.csv")
print("   Plots : plots/fairness_gender.png, fairness_region.png, fairness_disability.png")