"""Read-only regression checks for saved experiments and notebook code cells."""
from pathlib import Path
import hashlib
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from src.data_engine import FEATURES, PROFILE_FILE, COURSE_FILE, MODEL_DIR, MODEL_FILE
from src.experiments import candidates, choose_threshold, scores, split_dataset, preprocessor


def main():
    profiles, data = pd.read_csv(PROFILE_FILE), pd.read_csv(COURSE_FILE)
    meta = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(MODEL_DIR / "metrics.csv")
    predictions = pd.read_csv(MODEL_DIR / "test_predictions.csv")
    validation = pd.read_csv(MODEL_DIR / "validation_predictions.csv")
    assert set(metrics.Model) == set(candidates()) and len(metrics) == 6
    assert metrics.Selected.sum() == 1
    assert not {"MSSV", "HoTen", "DiemCuoiKy", "NguyCo"} & set(FEATURES)
    assert hashlib.sha256(data.to_csv(index=False).encode()).hexdigest() == meta["data_sha256"]
    sets = split_dataset(profiles, data)
    saved_splits = pd.read_csv(MODEL_DIR / "student_splits.csv")
    for name, subset in sets.items():
        assert set(saved_splits.loc[saved_splits.Split.eq(name), "MSSV"]) == set(subset.MSSV)
    folds = pd.read_csv(MODEL_DIR / "cv_folds.csv")
    assert folds.MSSV.is_unique
    assert set(folds.MSSV) == set(sets["train"].MSSV)
    assert folds.HeldOutFold.nunique() == meta["cv_folds"]
    winner = metrics.loc[metrics.Model.ne("Dummy Baseline")].sort_values("CV_AP_mean", ascending=False).iloc[0]
    assert winner.Model == meta["selected_model"]
    assert np.isclose(winner.Threshold, meta["threshold"])
    for row in metrics.itertuples():
        pred = predictions.loc[predictions.Model.eq(row.Model)]
        val = validation.loc[validation.Model.eq(row.Model)]
        assert set(pred.MSSV) == set(sets["test"].MSSV)
        assert not set(pred.MSSV) & set(val.MSSV)
        assert len(pred) == len(sets["test"])
        for key, value in scores(pred.NguyCo, pred.Probability, row.Threshold).items():
            assert np.isclose(value, getattr(row, key)), (row.Model, key)
        threshold, _ = choose_threshold(val.NguyCo, val.Probability)
        assert np.isclose(threshold, row.Threshold)
        model = joblib.load(MODEL_DIR / meta["model_files"][row.Model])
        assert np.allclose(model.predict_proba(pred[FEATURES])[:, 1], pred.Probability)
    production = joblib.load(MODEL_FILE)
    selected = predictions.loc[predictions.Model.eq(meta["selected_model"])]
    assert np.allclose(production.predict_proba(selected[FEATURES])[:, 1], selected.Probability)
    # Missing numeric values and unseen categories do not leak or crash preprocessing.
    sample = sets["train"][FEATURES].head(20).copy()
    sample.loc[sample.index[0], "GPA_TichLuy"] = np.nan
    pre = preprocessor().fit(sample)
    unseen = sample.iloc[[0]].copy()
    unseen["Khoa"] = "Khoa chưa gặp"
    assert np.isfinite(pre.transform(unseen)).all()
    duplicate = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    try:
        split_dataset(profiles, duplicate)
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate guard failed")
    assert MODEL_FILE.exists()
    print("PASS: six models, group isolation, CV manifest, metrics, validation thresholds, model predictions, missing/unseen inputs")


if __name__ == "__main__":
    main()
