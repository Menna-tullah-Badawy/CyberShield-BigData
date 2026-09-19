"""
Feature Store Manager (.npy track).
منطق Cell 0/Cell 1: حفظ وتحميل السلاسل + الـ metadata (بما فيها الأسماء الحقيقية).
"""

import json
import os
from typing import Dict, Mapping

import numpy as np

NPY_FILES = ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test")
METADATA_FILE = "strict_temporal_metadata.json"


class FeatureStoreManager:
    """إدارة تخزين السلاسل الزمنية بصيغة .npy (نفس مسارات النوت بوك)."""

    def __init__(self, base_path: str = "data/feature_store/nids_features_latest"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def save_sequences(
        self, arrays: Mapping[str, np.ndarray], feature_names: list, seq_len: int
    ) -> Dict[str, object]:
        """حفظ المصفوفات الست + الـ metadata وإرجاعها."""
        for name in NPY_FILES:
            np.save(os.path.join(self.base_path, f"{name}.npy"), arrays[name])
        meta = {
            "sequence_length": seq_len,
            "final_feature_dimension": int(arrays["X_train"].shape[2]),
            "train_shape": list(arrays["X_train"].shape),
            "val_shape": list(arrays["X_val"].shape),
            "test_shape": list(arrays["X_test"].shape),
            "feature_names": list(feature_names),
        }
        with open(os.path.join(self.base_path, METADATA_FILE), "w") as f:
            json.dump(meta, f, indent=2)
        return meta

    def load_sequences(self):
        """تحميل الـ .npy الستة مع تنظيف NaN/Inf (نفس كود النوت بوك)."""
        from src.cleaning.cleaning_pipeline import CleaningPipeline

        arrays = []
        for name in NPY_FILES:
            arr = np.load(os.path.join(self.base_path, f"{name}.npy"))
            if name.startswith("X_"):
                arr = CleaningPipeline.sanitize_array(arr)
            arrays.append(arr)
        return tuple(arrays)

    def load_metadata(self) -> Dict[str, object]:
        with open(os.path.join(self.base_path, METADATA_FILE)) as f:
            return json.load(f)

    def exists(self) -> bool:
        """هل الـ Store مبني وجاهز؟ (لتجاوز المراحل المكتملة)."""
        return all(os.path.exists(os.path.join(self.base_path, f"{n}.npy")) for n in NPY_FILES)
