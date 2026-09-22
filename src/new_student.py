"""Validation and frozen-model evaluation for user-supplied new students.

No writes, no fit, and no threshold tuning in this module.
"""
import csv
import hashlib
from io import StringIO
import re

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from src.data_engine import FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES

# Application input limits, not claims about university regulations.
NUMERIC_INPUTS = {
    "GPA_TichLuy": ("GPA tích luỹ /4", 0., 4., 2.7, .01),
    "TinChiTichLuy": ("Tín chỉ đã tích luỹ", 0., 300., 80., 1.),
    "DiemQuaTrinh": ("Điểm quá trình /10", 0., 10., 6.5, .1),
    "ChuyenCan": ("Chuyên cần (%)", 0., 100., 85., .1),
    "TruyCapLMS": ("Truy cập LMS /tuần", 0., 1000., 12., 1.),
    "TyLeNopDungHan": ("Bài nộp đúng hạn (%)", 0., 100., 85., .1),
    "DiemRenLuyen": ("Điểm rèn luyện /100", 0., 100., 75., 1.),
    "GioTuHoc": ("Tự học (giờ/tuần)", 0., 168., 7., .5),
    "TinChi": ("Tín chỉ học phần", 1., 30., 3., 1.),
}
INTEGER_FIELDS = {"TinChiTichLuy", "TruyCapLMS", "DiemRenLuyen", "TinChi"}
MAX_ROWS = 10000
MAX_BYTES = 2 * 1024 * 1024


def parse_csv(payload):
    if len(payload) > MAX_BYTES:
        raise ValueError("CSV tối đa 2 MB.")
    try:
        text = payload.decode("utf-8-sig")
        first = text.splitlines()[0]
        delimiter = ";" if first.count(";") > first.count(",") else ","
        csv_rows = csv.reader(StringIO(text), delimiter=delimiter, strict=True)
        header = [x.strip() for x in next(csv_rows)]
        if len(set(header)) != len(header):
            raise ValueError("CSV có tên cột trùng nhau.")
        for number, row in enumerate(csv_rows, 2):
            if row and len(row) != len(header):
                raise ValueError(f"Dòng CSV {number}: số ô không khớp tiêu đề.")
        frame = pd.read_csv(StringIO(text), sep=delimiter, dtype=str, keep_default_na=False, nrows=MAX_ROWS + 1)
        frame.columns = header
        return frame
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError, IndexError, StopIteration, csv.Error) as exc:
        raise ValueError("Không đọc được CSV. Dùng UTF-8, dấu phẩy/chấm phẩy và dấu chấm cho số thập phân.") from exc


def feature_fingerprints(frame):
    normalized = frame[FEATURES].copy()
    for field in NUMERIC_FEATURES:
        normalized[field] = pd.to_numeric(normalized[field]).astype(float)
    for field in CATEGORICAL_FEATURES:
        normalized[field] = normalized[field].astype(str).str.strip()
    return pd.util.hash_pandas_object(normalized, index=False)


def validate_new_students(frame, reference):
    """Return clean data, errors, warnings, independence flag. No silent row drops."""
    errors, warnings = [], []
    data = frame.copy().reset_index(drop=True)
    required = ["MSSV", "MaHocPhan"] + FEATURES
    missing = sorted(set(required) - set(data.columns))
    if missing:
        return data, ["Thiếu cột: " + ", ".join(missing)], [], False
    if not 1 <= len(data) <= MAX_ROWS:
        return data, [f"Cần từ 1 đến {MAX_ROWS:,} dòng."], [], False
    if data.columns.duplicated().any():
        return data, ["Tên cột bị trùng."], [], False
    for field in ["MSSV", "MaHocPhan"] + CATEGORICAL_FEATURES:
        if data[field].isna().any() or data[field].astype(str).str.strip().eq("").any():
            errors.append(f"{field}: không được để trống.")
        data[field] = data[field].astype(str).str.strip()
    for field in ["MSSV", "MaHocPhan"]:
        data[field] = data[field].str.upper()
        if not data[field].str.fullmatch(r"[A-Z0-9_-]{1,40}").all():
            errors.append(f"{field}: dùng 1–40 ký tự chữ không dấu, số, _ hoặc -.")
    for field, (label, low, high, *_rest) in NUMERIC_INPUTS.items():
        values = pd.to_numeric(data[field], errors="coerce")
        valid = values.notna() & np.isfinite(values) & values.between(low, high)
        if field in INTEGER_FIELDS:
            valid &= values.mod(1).eq(0)
        if not valid.all():
            rows = ", ".join(str(i + 2) for i in data.index[~valid][:8])
            errors.append(f"{label}: cần {'số nguyên' if field in INTEGER_FIELDS else 'số'} từ {low:g} đến {high:g}; dòng CSV {rows}.")
        data[field] = values
    if data.duplicated(["MSSV", "MaHocPhan"]).any():
        errors.append("Trùng MSSV–MaHocPhan. Không đếm nhiều lần cùng một lượt học phần.")
    known_ids = set(reference.MSSV.astype(str).str.strip().str.upper())
    if data.MSSV.isin(known_ids).any():
        errors.append("MSSV đã nằm trong dataset hiện tại (train/validation/test). Trang này chỉ dành cho sinh viên mới.")
    # Ground truth is optional; if a column is provided it must be complete.
    grade = None
    if "DiemCuoiKy" in data:
        grade = pd.to_numeric(data.DiemCuoiKy, errors="coerce")
        if not (grade.notna() & np.isfinite(grade) & grade.between(0, 10)).all():
            errors.append("DiemCuoiKy: cần đủ điểm thực tế 0–10 cho mọi dòng; nếu chưa có, bỏ cột này.")
        data["DiemCuoiKy"] = grade
    if "NguyCo" in data:
        labels = pd.to_numeric(data.NguyCo, errors="coerce")
        if not labels.isin([0, 1]).all():
            errors.append("NguyCo: chỉ nhận 0 hoặc 1, không để thiếu nhãn giữa các dòng.")
        if grade is not None and not labels.eq(grade.lt(5).astype(int)).all():
            errors.append("NguyCo mâu thuẫn DiemCuoiKy: nhãn 1 phải tương ứng điểm dưới 5.")
        data["NguyCo"] = labels
    elif grade is not None:
        data["NguyCo"] = grade.lt(5).astype(int)
    if errors:
        return data, errors, warnings, False
    for field in NUMERIC_FEATURES:
        outside = ~data[field].between(reference[field].min(), reference[field].max())
        if outside.any():
            warnings.append(f"{field}: {outside.sum()} dòng ngoài khoảng quan sát của dataset; dự đoán có thể kém tin cậy.")
    for field in CATEGORICAL_FEATURES:
        unseen = sorted(set(data[field]) - set(reference[field].astype(str).str.strip()))
        if unseen:
            warnings.append(f"{field}: danh mục chưa gặp ({', '.join(unseen[:3])}); encoder bỏ qua danh mục mới, không đồng nghĩa đã học được nó.")
    catalog = reference.drop_duplicates("MaHocPhan").set_index("MaHocPhan")
    for row in data[["MaHocPhan", "HocPhan", "TinChi"]].drop_duplicates().itertuples(index=False):
        if row.MaHocPhan in catalog.index:
            known = catalog.loc[row.MaHocPhan]
            if row.HocPhan != known.HocPhan or row.TinChi != known.TinChi:
                errors.append(f"{row.MaHocPhan}: tên học phần hoặc tín chỉ không khớp danh mục hiện tại.")
    duplicate_features = feature_fingerprints(data).isin(set(feature_fingerprints(reference)))
    independent = not duplicate_features.any() and not errors
    if duplicate_features.any():
        warnings.append(f"{duplicate_features.sum()} dòng có toàn bộ đặc trưng giống dataset hiện tại. Chỉ dự đoán; không tính là kiểm thử độc lập bằng cách đổi MSSV.")
    return data, errors, warnings, independent


def predict_new_students(model, frame, metadata):
    result = frame.copy()
    # Strict feature allow-list: the supplied outcome NEVER enters predict_proba.
    positive = list(model.classes_).index(1)
    result["Probability"] = model.predict_proba(result[FEATURES])[:, positive]
    result["Threshold"] = float(metadata["threshold"])
    result["Prediction"] = result.Probability.ge(result.Threshold).astype(int)
    result["Model"] = metadata["selected_model"]
    result["RunID"] = metadata["run_id"]
    if "NguyCo" in result:
        result["Correct"] = result.Prediction.eq(result.NguyCo)
        result["Error"] = np.where(result.NguyCo.eq(1), np.where(result.Prediction.eq(1), "TP", "FN"),
                                   np.where(result.Prediction.eq(1), "FP", "TN"))
    return result


def evaluate_new_students(result):
    if "NguyCo" not in result or result.empty:
        return None
    y = result.NguyCo.astype(int)
    tn, fp, fn, tp = [int(x) for x in confusion_matrix(y, result.Prediction, labels=[0, 1]).ravel()]
    both = y.nunique() == 2
    return {"N": len(result), "Students": int(result.MSSV.nunique()), "Positive": int(y.sum()),
            "Accuracy": (tp+tn)/len(result),
            "Precision": tp/(tp+fp) if tp+fp else None,
            "Recall": tp/(tp+fn) if tp+fn else None,
            "F1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            "ROC_AUC": float(roc_auc_score(y, result.Probability)) if both else None,
            "AP": float(average_precision_score(y, result.Probability)) if both else None,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def template_csv(reference, labelled=False):
    course = reference.drop_duplicates("MaHocPhan").iloc[0]
    row = {"MSSV": "NEW_DEMO_001", "MaHocPhan": course.MaHocPhan,
           **{field: spec[3] for field, spec in NUMERIC_INPUTS.items()},
           "Khoa": str(reference.Khoa.iloc[0]), "HocPhan": course.HocPhan,
           "NhomHoc": str(reference.NhomHoc.iloc[0]), "TinChi": int(course.TinChi)}
    columns = ["MSSV", "MaHocPhan"] + FEATURES
    if labelled:
        row["DiemCuoiKy"] = 4.5
        columns.append("DiemCuoiKy")
    return pd.DataFrame([row])[columns].to_csv(index=False).encode("utf-8-sig")


def input_digest(frame):
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


def safe_csv(frame):
    """Prevent spreadsheet formula interpretation of user-supplied text on export."""
    exported = frame.copy()
    for column in exported.select_dtypes(include=["object", "string"]).columns:
        exported[column] = exported[column].map(lambda x: "'" + x if isinstance(x, str) and re.match(r"^[\s]*[=+@-]", x) else x)
    return exported.to_csv(index=False).encode("utf-8-sig")
