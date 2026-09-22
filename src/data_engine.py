"""Synthetic academic data, training, and prediction services for EduPredict AI."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = ROOT / "data" / "generated"
MODEL_DIR = ROOT / "results" / "academic_model"
PROFILE_FILE = GENERATED_DIR / "student_profiles.csv"
COURSE_FILE = GENERATED_DIR / "course_records.csv"
MODEL_FILE = MODEL_DIR / "risk_model_v2_4.pkl"
METRICS_FILE = MODEL_DIR / "metrics.csv"
VERSION_FILE = GENERATED_DIR / ".dataset_version"
DATASET_VERSION = "2.7"

NUMERIC_FEATURES = [
    "GPA_TichLuy", "TinChiTichLuy", "DiemQuaTrinh", "ChuyenCan",
    "TruyCapLMS", "TyLeNopDungHan", "DiemRenLuyen", "GioTuHoc", "TinChi",
]
CATEGORICAL_FEATURES = ["Khoa", "HocPhan", "NhomHoc"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _clip(value: float, low: float, high: float) -> float:
    return round(float(np.clip(value, low, high)), 1)


def build_demo_dataset(student_count: int = 1500, seed: int = 2025) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a transparent, reproducible university-semester demo dataset."""
    rng = np.random.default_rng(seed)
    courses = [
        ("IT4010", "Học máy & Khai phá dữ liệu", 3),
        ("IT4020", "Hệ quản trị CSDL nâng cao", 3),
        ("IT4030", "Phát triển ứng dụng Web", 4),
        ("MA2010", "Xác suất thống kê ứng dụng", 3),
    ]
    first_names = ["An", "Bình", "Chi", "Dũng", "Giang", "Hà", "Hải", "Hân", "Hương", "Khánh", "Lan", "Long", "Linh", "Minh", "My", "Nam", "Ngọc", "Phúc", "Quân", "Trang", "Vy"]
    surnames = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Vũ", "Đặng", "Bùi", "Đỗ", "Hồ"]
    profiles: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    for index in range(student_count):
        student_id = f"SV{20210001 + index:08d}"
        name = f"{rng.choice(surnames)} {rng.choice(first_names)} {rng.choice(first_names)}"
        faculty = str(rng.choice(["Công nghệ thông tin", "Kinh tế số", "Kỹ thuật dữ liệu"], p=[0.58, 0.24, 0.18]))
        cohort = f"K{64 + index % 3}"
        latent = rng.normal(0, 1)
        gpa = _clip(2.7 + 0.47 * latent + rng.normal(0, 0.2), 1.2, 3.95)
        credits = int(np.clip(80 + 19 * latent + rng.normal(0, 14), 28, 132))
        profiles.append({
            "MSSV": student_id, "HoTen": name, "Khoa": faculty, "KhoaHoc": cohort,
            "GPA_TichLuy": gpa, "TinChiTichLuy": credits,
            "XepLoai": "Xuất sắc" if gpa >= 3.6 else "Khá" if gpa >= 3.2 else "Khá tích cực" if gpa >= 2.5 else "Cần cải thiện",
        })

        for code, course, credit in courses:
            attendance = _clip(83 + latent * 7 + rng.normal(0, 9), 45, 100)
            lms_visits = int(np.clip(11 + latent * 3 + rng.normal(0, 4), 1, 25))
            on_time = _clip(84 + latent * 6 + rng.normal(0, 10), 35, 100)
            conduct = int(np.clip(75 + latent * 6 + rng.normal(0, 8), 50, 100))
            self_study = _clip(6.5 + latent * 1.6 + rng.normal(0, 1.8), 0, 16)
            course_effect = {"IT4020": -0.45, "IT4010": -0.12, "IT4030": 0.25, "MA2010": -0.2}[code]
            process_score = _clip(5.4 + latent * 0.9 + attendance * 0.018 + on_time * 0.008 + course_effect + rng.normal(0, 0.8), 0, 10)
            # The intercept intentionally yields a meaningful early-warning
            # cohort while preserving a majority of passing observations.
            final_score = _clip(0.9 + process_score * 0.57 + attendance * 0.012 + self_study * 0.11 + course_effect + rng.normal(0, 1.05), 0, 10)
            records.append({
                "MSSV": student_id, "MaHocPhan": code, "HocPhan": course, "TinChi": credit,
                "NhomHoc": f"Nhóm {index % 4 + 1:02d}", "Khoa": faculty,
                "GPA_TichLuy": gpa, "TinChiTichLuy": credits,
                "DiemQuaTrinh": process_score, "ChuyenCan": attendance,
                "TruyCapLMS": lms_visits, "TyLeNopDungHan": on_time,
                "DiemRenLuyen": conduct, "GioTuHoc": self_study,
                "DiemCuoiKy": final_score, "NguyCo": int(final_score < 5.0),
            })

    return pd.DataFrame(profiles), pd.DataFrame(records)


def train_and_save(profiles: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    """Compatibility wrapper for the grouped validation/test pipeline."""
    from src.training import train
    return train(profiles, records)


def ensure_demo_assets() -> None:
    # Never replace an existing academic CSV merely because a version changed.
    if PROFILE_FILE.exists() and COURSE_FILE.exists():
        if not (MODEL_FILE.exists() and METRICS_FILE.exists() and (MODEL_DIR / "metadata.json").exists()):
            from src.training import train
            train(pd.read_csv(PROFILE_FILE), pd.read_csv(COURSE_FILE))
        return
    if PROFILE_FILE.exists() or COURSE_FILE.exists():
        raise ValueError("Thiếu một trong hai CSV học vụ; cần khôi phục file còn thiếu.")
    profiles, records = build_demo_dataset()
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(PROFILE_FILE, index=False, encoding="utf-8-sig")
    records.to_csv(COURSE_FILE, index=False, encoding="utf-8-sig")
    from src.training import train
    train(profiles, records)
    VERSION_FILE.write_text(DATASET_VERSION, encoding="utf-8")


def load_assets() -> tuple[pd.DataFrame, pd.DataFrame, Pipeline, pd.DataFrame]:
    ensure_demo_assets()
    records = pd.read_csv(COURSE_FILE)
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    fingerprint = hashlib.sha256(records.to_csv(index=False).encode()).hexdigest()
    if metadata.get("data_sha256") != fingerprint:
        raise ValueError("CSV đã thay đổi sau lần huấn luyện. Chạy python -m src.training để đồng bộ mô hình và chỉ số.")
    return (
        pd.read_csv(PROFILE_FILE),
        records,
        joblib.load(MODEL_FILE),
        pd.read_csv(METRICS_FILE),
    )


def predict_risk(model: Pipeline, record: pd.Series | dict[str, Any]) -> float:
    frame = pd.DataFrame([record])[FEATURES]
    return float(model.predict_proba(frame)[0, 1])


def predict_risks(model: Pipeline, records: pd.DataFrame) -> pd.Series:
    """Batch prediction for dashboards; avoids one model invocation per row."""
    return pd.Series(model.predict_proba(records[FEATURES])[:, 1], index=records.index)
