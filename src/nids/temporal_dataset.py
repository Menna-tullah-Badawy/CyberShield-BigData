"""
Temporal Feature Extraction & Leakage-Proof Split.
نقل حرفي من Cell 0 في cybershield.ipynb.

Pipeline:
    CICIDS-2018 CSVs (day per file) → timestamp sort → 1:5 sampling →
    binary target → STRICT_EXCLUDE filter → per-day temporal split (70/15/15) →
    day sequences (SEQ_LEN=10, attack vote >= 0.2) →
    zero-variance filter (fit on train ONLY) → .npy + metadata.json
"""

import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── ثوابت النوت بوك بالنص ──────────────────────────────────────────────
SEQ_LEN = 10
SAMPLE_STEP = 5          # عينة كل 5 صفوف (تمثيل 20% عبر الـ 24 ساعة)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.85         # أي: train 70% / val 15% / test 15%
SEQ_ATTACK_VOTE = 0.2    # السلسلة Attack لو متوسط الليبل >= 0.2
ZERO_VAR_EPS = 1e-5
KAGGLE_DATASET_ID = "solarmainframe/ids-intrusion-csv"

# قائمة الحظر الصارمة لمنع تسريب الهوية والمنافذ (بالنص من النوت بوك)
STRICT_EXCLUDE = [
    'label', 'target', 'timestamp', 'ts_parsed', 'flow id',
    'src ip', 'dst ip', 'source ip', 'destination ip',
    'src port', 'dst port', 'source port', 'destination port',
    'protocol', 'unnamed: 0',
]

NPY_FILES = ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test")
METADATA_FILE = "strict_temporal_metadata.json"


def create_day_sequences(
    X_mat: np.ndarray, y_arr: np.ndarray, seq_len: int = SEQ_LEN
) -> Tuple[np.ndarray, np.ndarray]:
    """تقطيع مصفوفة اليوم لسلاسل زمنية (نفس دالة النوت بوك)."""
    num_seq = len(X_mat) // seq_len
    if num_seq == 0:
        return np.empty((0, seq_len, X_mat.shape[1])), np.empty((0,))
    X_cut = X_mat[:num_seq * seq_len].reshape(num_seq, seq_len, X_mat.shape[1])
    y_cut = y_arr[:num_seq * seq_len].reshape(num_seq, seq_len)
    y_final = (np.mean(y_cut, axis=1) >= SEQ_ATTACK_VOTE).astype(np.int64)
    return X_cut, y_final


def process_day_csv(
    file_path: str,
    seq_len: int = SEQ_LEN,
    sample_step: int = SAMPLE_STEP,
) -> Optional[Tuple[Tuple[np.ndarray, np.ndarray],
                     Tuple[np.ndarray, np.ndarray],
                     Tuple[np.ndarray, np.ndarray],
                     List[str], int]]:
    """معالجة ملف يوم واحد: ترتيب زمني → عينة → Target → تقسيم → سلاسل."""
    df_day = pd.read_csv(file_path, low_memory=False)
    df_day.columns = df_day.columns.str.strip()

    ts_col = next((c for c in df_day.columns if 'timestamp' in c.lower()), None)
    lbl_col = next((c for c in df_day.columns if 'label' in c.lower()), None)
    if lbl_col is None:
        raise ValueError(f"No label column in {file_path}")

    if ts_col:
        df_day["ts_parsed"] = pd.to_datetime(df_day[ts_col], errors="coerce", dayfirst=True)
        df_day = df_day.dropna(subset=["ts_parsed"]).sort_values("ts_parsed").reset_index(drop=True)

    # أخذ عينة زمنية ممتدة على مدار الـ 24 ساعة
    df_day = df_day.iloc[::sample_step].reset_index(drop=True)

    df_day['target'] = df_day[lbl_col].apply(
        lambda x: 0 if str(x).strip().lower() == 'benign' else 1)

    # استبعاد المنافذ والعناوين التعريفية
    valid_cols = [c for c in df_day.columns if c.lower() not in STRICT_EXCLUDE]
    df_day[valid_cols] = df_day[valid_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)

    # تقسيم اليوم الواحد زمنياً
    n = len(df_day)
    t_end, v_end = int(n * TRAIN_RATIO), int(n * VAL_RATIO)

    df_tr = df_day.iloc[:t_end]
    df_va = df_day.iloc[t_end:v_end]
    df_te = df_day.iloc[v_end:]

    # بناء السلاسل الزمنية لكل جزء بشكل مستقل
    x_tr, y_tr = create_day_sequences(
        df_tr[valid_cols].to_numpy(dtype=np.float32),
        df_tr['target'].to_numpy(dtype=np.int64), seq_len)
    x_va, y_va = create_day_sequences(
        df_va[valid_cols].to_numpy(dtype=np.float32),
        df_va['target'].to_numpy(dtype=np.int64), seq_len)
    x_te, y_te = create_day_sequences(
        df_te[valid_cols].to_numpy(dtype=np.float32),
        df_te['target'].to_numpy(dtype=np.int64), seq_len)

    attacks = int((df_day['target'] == 1).sum())
    return (x_tr, y_tr), (x_va, y_va), (x_te, y_te), valid_cols, attacks


def build_temporal_dataset(
    csv_dir: Optional[str] = None,
    output_dir: str = "data/feature_store/nids_features_latest",
    seq_len: int = SEQ_LEN,
    sample_step: int = SAMPLE_STEP,
    kaggle_dataset_id: str = KAGGLE_DATASET_ID,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    بناء الداتاسيت الزمنية كاملة من ملفات CSV.

    إن لم يُمرر csv_dir على Kaggle يتم تنزيل الداتاسيت عبر kagglehub
    (نفس سلوك النوت بوك). يُحفظ .npy + metadata (مضاف لها feature_names
    الحقيقية لإصلاح عدم تطابق أسماء XAI في النوت بوك الأصلية).
    """
    if csv_dir is None:
        try:
            import kagglehub
        except ImportError:
            os.system("pip install -q kagglehub")
            import kagglehub
        if verbose:
            print("\n>>> Downloading CICIDS 2018 dataset...")
        csv_dir = kagglehub.dataset_download(kaggle_dataset_id)

    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not csv_files:
        csv_files = glob.glob(os.path.join(csv_dir, "**", "*.csv"), recursive=True)
    if verbose:
        print(f"📅 Found {len(csv_files)} days of traffic.")

    train_seq_x, train_seq_y = [], []
    val_seq_x, val_seq_y = [], []
    test_seq_x, test_seq_y = [], []
    feature_names: List[str] = []

    for file in sorted(csv_files):
        day_name = os.path.basename(file)
        try:
            (x_tr, y_tr), (x_va, y_va), (x_te, y_te), valid_cols, attacks = process_day_csv(
                file, seq_len=seq_len, sample_step=sample_step)
            if not feature_names:
                feature_names = list(valid_cols)
            if len(x_tr) > 0:
                train_seq_x.append(x_tr); train_seq_y.append(y_tr)
            if len(x_va) > 0:
                val_seq_x.append(x_va); val_seq_y.append(y_va)
            if len(x_te) > 0:
                test_seq_x.append(x_te); test_seq_y.append(y_te)
            if verbose:
                print(f"   ✅ Processed {day_name} | Attacks: {attacks}")
        except Exception as e:
            if verbose:
                print(f"   ⚠️ Skipped {day_name}: {e}")

    X_train = np.concatenate(train_seq_x, axis=0)
    y_train = np.concatenate(train_seq_y, axis=0)
    X_val = np.concatenate(val_seq_x, axis=0)
    y_val = np.concatenate(val_seq_y, axis=0)
    X_test = np.concatenate(test_seq_x, axis=0)
    y_test = np.concatenate(test_seq_y, axis=0)

    # تصفية الميزات عديمة التباين بناءً على بيانات التدريب فقط لمنع الـ Leakage
    stds = np.std(X_train.reshape(-1, X_train.shape[2]), axis=0)
    valid_feats = stds > ZERO_VAR_EPS

    X_train = X_train[:, :, valid_feats]
    X_val = X_val[:, :, valid_feats]
    X_test = X_test[:, :, valid_feats]
    kept_names = [n for n, k in zip(feature_names, valid_feats) if k]

    os.makedirs(output_dir, exist_ok=True)
    for name, arr in [("X_train", X_train), ("y_train", y_train),
                      ("X_val", X_val), ("y_val", y_val),
                      ("X_test", X_test), ("y_test", y_test)]:
        np.save(os.path.join(output_dir, f"{name}.npy"), arr)

    meta = {
        "sequence_length": seq_len,
        "final_feature_dimension": int(X_train.shape[2]),
        "train_shape": list(X_train.shape),
        "val_shape": list(X_val.shape),
        "test_shape": list(X_test.shape),
        "feature_names": kept_names,  # ← إصلاح: أسماء الميزات الحقيقية المحفوظة
    }
    with open(os.path.join(output_dir, METADATA_FILE), "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print("\n" + "=" * 50)
        print(f"🟢 Clean Train Shape: {X_train.shape} | Attacks: {(y_train == 1).sum()}")
        print(f"🔵 Clean Val Shape:   {X_val.shape}   | Attacks: {(y_val == 1).sum()}")
        print(f"🟠 Clean Test Shape:  {X_test.shape}  | Attacks: {(y_test == 1).sum()}")
        print(f"📊 Safe Feature Count: {X_train.shape[2]} (Zero Ports, Zero Metadata)")
        print("=" * 50)
    return meta


def load_sequences(store_dir: str) -> Tuple[np.ndarray, np.ndarray,
                                            np.ndarray, np.ndarray,
                                            np.ndarray, np.ndarray]:
    """تحميل الـ .npy الستة مع تنظيف NaN/Inf (نفس كود النوت بوك)."""
    arrays = []
    for name in NPY_FILES:
        arr = np.load(os.path.join(store_dir, f"{name}.npy"))
        if name.startswith("X_"):
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arrays.append(arr)
    return tuple(arrays)  # type: ignore[return-value]


def load_metadata(store_dir: str) -> Dict[str, object]:
    with open(os.path.join(store_dir, METADATA_FILE)) as f:
        return json.load(f)
