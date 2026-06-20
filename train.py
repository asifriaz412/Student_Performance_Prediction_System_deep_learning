"""
Student Performance Prediction System Using Deep Learning
OULAD Dataset – Final Year Project
Complete, robust pipeline with model & plot saving.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')           # Non-interactive backend – no pop-ups
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, Input, LSTM, GRU,
                                     Bidirectional, Conv1D, MaxPooling1D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# ------------------------- CONFIGURATION -------------------------
DATA_PATH = "data/"
MODEL_DIR = "models"
PLOT_DIR  = "plots"
TARGET = "final_result"
TEST_SIZE = 0.2
VAL_SPLIT = 0.2
EPOCHS = 50
BATCH_SIZE = 32
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ------------------------- 1. LOAD DATA -------------------------
print("Loading data...")
info = pd.read_csv(DATA_PATH + "studentInfo.csv")
assessment = pd.read_csv(DATA_PATH + "studentAssessment.csv")
vle = pd.read_csv(DATA_PATH + "studentVle.csv")
assessments = pd.read_csv(DATA_PATH + "assessments.csv")
courses = pd.read_csv(DATA_PATH + "courses.csv")

# ---------- 1.1 Fix data types ----------
assessment['score'] = pd.to_numeric(assessment['score'], errors='coerce')
vle['sum_click'] = pd.to_numeric(vle['sum_click'], errors='coerce')

# ------------------------- 2. STATIC PREPROCESSING ---------------
print("Preprocessing static features...")

# Merge assessment with assessments to get code_module/presentation
assessment_with_course = assessment.merge(
    assessments[['id_assessment', 'code_module', 'code_presentation']],
    on='id_assessment', how='left'
)

# Aggregate assessments
assess_agg = assessment_with_course.groupby(
    ['id_student', 'code_module', 'code_presentation']
).agg(
    avg_score=('score', 'mean'),
    submission_count=('id_assessment', 'count')
).reset_index()

# Aggregate VLE data
vle_agg = vle.groupby(
    ['id_student', 'code_module', 'code_presentation']
).agg(
    total_clicks=('sum_click', 'sum'),
    active_days=('date', 'nunique')
).reset_index()
vle_agg['engagement_ratio'] = vle_agg['total_clicks'] / vle_agg['active_days'].replace(0, np.nan)

# Merge into studentInfo
df = info.merge(assess_agg, on=['id_student', 'code_module', 'code_presentation'],
                how='left')
df = df.merge(vle_agg, on=['id_student', 'code_module', 'code_presentation'],
              how='left')

# Fill missing aggregated features
df['avg_score'] = df['avg_score'].fillna(0)
df['submission_count'] = df['submission_count'].fillna(0)
df['total_clicks'] = df['total_clicks'].fillna(0)
df['active_days'] = df['active_days'].fillna(0)
df['engagement_ratio'] = df['engagement_ratio'].fillna(0)

# Categorical encoding
cat_cols = ['gender', 'region', 'highest_education', 'imd_band',
            'age_band', 'disability']
label_encoders = {}
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Binary target
df[TARGET] = df[TARGET].map({'Pass':1, 'Distinction':1, 'Fail':0, 'Withdrawn':0})
df = df.dropna(subset=[TARGET])
df[TARGET] = df[TARGET].astype(int)

# Feature set for ANN
static_feature_cols = ['gender', 'region', 'highest_education', 'imd_band',
                       'age_band', 'disability', 'num_of_prev_attempts',
                       'studied_credits', 'avg_score', 'submission_count',
                       'total_clicks', 'active_days', 'engagement_ratio']
X_static = df[static_feature_cols].copy()
y = df[TARGET].values

# Scale numeric columns
num_cols = ['num_of_prev_attempts', 'studied_credits', 'avg_score',
            'submission_count', 'total_clicks', 'active_days', 'engagement_ratio']
scaler = StandardScaler()
X_static[num_cols] = scaler.fit_transform(X_static[num_cols])

# Keep a copy for sequence creation
df_static_encoded = df.copy()

# ------------------------- 3. SEQUENCE CREATION ---------------------
print("Creating sequential features...")

# Daily VLE data
vle_daily = vle.groupby(
    ['id_student', 'code_module', 'code_presentation', 'date']
).agg(
    sum_click=('sum_click', 'sum'),
    activity_count=('id_site', 'nunique')
).reset_index().rename(columns={'date': 'day'})

# Daily assessment scores – FIXED: include code_module/presentation
assess_dates = assessment.merge(
    assessments[['id_assessment', 'code_module', 'code_presentation', 'date']],
    on='id_assessment', how='left'
)
assess_dates = assess_dates[['id_student', 'code_module', 'code_presentation',
                             'date', 'score']].rename(columns={'date': 'day'})

# Demographics per instance (scaled)
demo_cols = ['gender', 'region', 'highest_education', 'imd_band',
             'age_band', 'disability', 'num_of_prev_attempts', 'studied_credits']
demo_scaler = StandardScaler()
df_static_encoded[['num_of_prev_attempts', 'studied_credits']] = demo_scaler.fit_transform(
    df_static_encoded[['num_of_prev_attempts', 'studied_credits']]
)
demo_dict = {}
for _, row in df_static_encoded.iterrows():
    key = (row['id_student'], row['code_module'], row['code_presentation'])
    demo_dict[key] = row[demo_cols].values.astype(np.float32)

# Build sequences
seq_list = []
y_seq_list = []
instance_keys = []

for key, vle_group in vle_daily.groupby(['id_student', 'code_module', 'code_presentation']):
    demo_vec = demo_dict.get(key)
    if demo_vec is None:
        continue

    # Assessment scores for this instance
    assess_instance = assess_dates[
        (assess_dates['id_student'] == key[0]) &
        (assess_dates['code_module'] == key[1]) &
        (assess_dates['code_presentation'] == key[2])
    ]
    score_dict = dict(zip(assess_instance['day'], assess_instance['score']))

    all_days = vle_group['day'].values
    if len(all_days) == 0:
        day_range = np.array([0])
    else:
        min_day, max_day = all_days.min(), all_days.max()
        day_range = np.arange(min_day, max_day + 1)

    sum_click_arr = np.zeros(len(day_range))
    activity_arr = np.zeros(len(day_range))
    score_arr = np.zeros(len(day_range))
    last_score = 0.0

    vle_day_map = vle_group.set_index('day')
    for i, day in enumerate(day_range):
        if day in vle_day_map.index:
            sum_click_arr[i] = vle_day_map.loc[day, 'sum_click']
            activity_arr[i] = vle_day_map.loc[day, 'activity_count']
        if day in score_dict:
            last_score = score_dict[day]
        score_arr[i] = last_score

    time_varying = np.stack([sum_click_arr, activity_arr, score_arr], axis=1)
    demo_repeated = np.tile(demo_vec, (len(day_range), 1))
    instance_seq = np.concatenate([time_varying, demo_repeated], axis=1)

    seq_list.append(instance_seq)
    y_seq_list.append(df_static_encoded.loc[
        (df_static_encoded['id_student'] == key[0]) &
        (df_static_encoded['code_module'] == key[1]) &
        (df_static_encoded['code_presentation'] == key[2]),
        TARGET
    ].values[0])
    instance_keys.append(key)

# Pad sequences
MAX_SEQ_LEN = max([seq.shape[0] for seq in seq_list])
n_features = seq_list[0].shape[1]
X_seq = np.zeros((len(seq_list), MAX_SEQ_LEN, n_features))
for i, seq in enumerate(seq_list):
    X_seq[i, :seq.shape[0], :] = seq

y_seq = np.array(y_seq_list, dtype=np.int32)

# ------------------------- 4. TRAIN/TEST SPLIT ----------------------
# 4.1 Full static data for ANN
X_static_full = X_static.values
y_full = df_static_encoded[TARGET].values

train_idx, test_idx, y_train_static, y_test_static = train_test_split(
    np.arange(len(y_full)), y_full,
    test_size=TEST_SIZE,
    stratify=y_full,
    random_state=RANDOM_STATE
)

X_train_static = X_static_full[train_idx]
X_test_static  = X_static_full[test_idx]

# 4.2 Overlapping instances for sequence models
keys_in_order = list(zip(df_static_encoded['id_student'],
                         df_static_encoded['code_module'],
                         df_static_encoded['code_presentation']))

seq_key_to_idx = {key: i for i, key in enumerate(instance_keys)}

# Mask of static rows that have a sequence
has_seq = np.array([key in seq_key_to_idx for key in keys_in_order])

X_static_seq = X_static_full[has_seq]
y_seq_full   = y_full[has_seq]

# Map from static subset to sequence indices
seq_indices_for_static = [seq_key_to_idx[keys_in_order[i]] for i in range(len(keys_in_order)) if has_seq[i]]
seq_indices_arr = np.array(seq_indices_for_static)

# Stratified split on the overlapping subset
train_seq_idx, test_seq_idx, y_train_seq, y_test_seq = train_test_split(
    np.arange(len(y_seq_full)), y_seq_full,
    test_size=TEST_SIZE,
    stratify=y_seq_full,
    random_state=RANDOM_STATE
)

X_train_seq = X_seq[seq_indices_arr[train_seq_idx]]
X_test_seq  = X_seq[seq_indices_arr[test_seq_idx]]

print(f"Static (ANN) train: {X_train_static.shape}, test: {X_test_static.shape}")
print(f"Sequence (LSTM/GRU) train: {X_train_seq.shape}, test: {X_test_seq.shape}")
print(f"Max sequence length: {MAX_SEQ_LEN}, features: {n_features}")

# ------------------------- 5. BUILD MODELS -------------------------
def build_ann(input_dim):
    inp = Input(shape=(input_dim,))
    x = Dense(128, activation='relu')(inp)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)
    return Model(inp, out)

def build_lstm(timesteps, features):
    inp = Input(shape=(timesteps, features))
    x = LSTM(64, return_sequences=True)(inp)
    x = Dropout(0.3)(x)
    x = LSTM(32)(x)
    x = Dropout(0.3)(x)
    x = Dense(16, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)
    return Model(inp, out)

def build_bilstm(timesteps, features):
    inp = Input(shape=(timesteps, features))
    x = Bidirectional(LSTM(64, return_sequences=True))(inp)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(32))(x)
    x = Dropout(0.3)(x)
    x = Dense(16, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)
    return Model(inp, out)

def build_gru(timesteps, features):
    inp = Input(shape=(timesteps, features))
    x = GRU(64, return_sequences=True)(inp)
    x = Dropout(0.3)(x)
    x = GRU(32)(x)
    x = Dropout(0.3)(x)
    x = Dense(16, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)
    return Model(inp, out)

def build_cnn_lstm(timesteps, features):
    inp = Input(shape=(timesteps, features))
    x = Conv1D(64, 3, activation='relu', padding='same')(inp)
    x = MaxPooling1D(2)(x)
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)
    x = LSTM(32)(x)
    x = Dropout(0.3)(x)
    x = Dense(16, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)
    return Model(inp, out)

# ------------------------- 6. TRAINING & EVALUATION ----------------
def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name):
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=10,
                               restore_best_weights=True, verbose=0)
    history = model.fit(
        X_train, y_train,
        validation_split=VAL_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )

    # Save model
    model_path = os.path.join(MODEL_DIR, f"{model_name}.keras")
    model.save(model_path)
    print(f"Model saved -> {model_path}")

    # Plot & save training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{model_name} Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{model_name}_history.png"))
    plt.close()

    # Predict & evaluate
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n--- {model_name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("Confusion Matrix:\n", cm)

    # Save confusion matrix
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fail/Withdrawn', 'Pass/Distinction'],
                yticklabels=['Fail/Withdrawn', 'Pass/Distinction'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(os.path.join(PLOT_DIR, f"{model_name}_cm.png"))
    plt.close()

    return {'model': model_name, 'accuracy': acc, 'precision': prec,
            'recall': rec, 'f1': f1}

# ------------------------- 7. RUN ALL MODELS -----------------------
results = []

print("\n===== Training ANN =====")
results.append(train_and_evaluate(
    build_ann(X_train_static.shape[1]), X_train_static, y_train_static,
    X_test_static, y_test_static, "ANN"))

print("\n===== Training LSTM =====")
results.append(train_and_evaluate(
    build_lstm(MAX_SEQ_LEN, n_features), X_train_seq, y_train_seq,
    X_test_seq, y_test_seq, "LSTM"))

print("\n===== Training Bi-LSTM =====")
results.append(train_and_evaluate(
    build_bilstm(MAX_SEQ_LEN, n_features), X_train_seq, y_train_seq,
    X_test_seq, y_test_seq, "Bi-LSTM"))

print("\n===== Training GRU =====")
results.append(train_and_evaluate(
    build_gru(MAX_SEQ_LEN, n_features), X_train_seq, y_train_seq,
    X_test_seq, y_test_seq, "GRU"))

print("\n===== Training CNN-LSTM =====")
results.append(train_and_evaluate(
    build_cnn_lstm(MAX_SEQ_LEN, n_features), X_train_seq, y_train_seq,
    X_test_seq, y_test_seq, "CNN-LSTM"))

# ------------------------- 8. MODEL COMPARISON --------------------
results_df = pd.DataFrame(results).set_index('model')
print("\n" + "="*50)
print("PERFORMANCE COMPARISON")
print("="*50)
print(results_df.round(4))
results_df.to_csv(os.path.join(MODEL_DIR, "model_comparison.csv"))

best = results_df['f1'].idxmax()
print(f"\nBest model (F1-score): {best}")

# ======================== SAVE PREPROCESSORS ========================
import pickle

PREPROC_DIR = "preprocessors"
os.makedirs(PREPROC_DIR, exist_ok=True)

with open(os.path.join(PREPROC_DIR, "static_scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
with open(os.path.join(PREPROC_DIR, "demo_scaler.pkl"), "wb") as f:
    pickle.dump(demo_scaler, f)
with open(os.path.join(PREPROC_DIR, "label_encoders.pkl"), "wb") as f:
    pickle.dump(label_encoders, f)
with open(os.path.join(PREPROC_DIR, "static_feature_cols.pkl"), "wb") as f:
    pickle.dump(static_feature_cols, f)
with open(os.path.join(PREPROC_DIR, "demo_cols.pkl"), "wb") as f:
    pickle.dump(demo_cols, f)
with open(os.path.join(PREPROC_DIR, "seq_params.pkl"), "wb") as f:
    pickle.dump({'max_seq_len': MAX_SEQ_LEN, 'n_features': n_features}, f)

print("✅ Preprocessors saved to 'preprocessors/'")
print("\n🎓 Training complete. You can now run App.py to serve predictions.")