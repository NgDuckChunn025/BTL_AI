"""Grouped CV experiments on existing synthetic CSVs; never rewrites data."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, fbeta_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from src.data_engine import FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, MODEL_DIR, MODEL_FILE


def scores(y, probability, threshold=0.5):
    pred = np.asarray(probability) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"Accuracy": float(accuracy_score(y, pred)),
            "Recall_NguyCo": float(recall_score(y, pred, zero_division=0)),
            "Precision_NguyCo": float(precision_score(y, pred, zero_division=0)),
            "F1_NguyCo": float(f1_score(y, pred, zero_division=0)),
            "F2_NguyCo": float(fbeta_score(y, pred, beta=2, zero_division=0)),
            "ROC_AUC": float(roc_auc_score(y, probability)),
            "Average_Precision": float(average_precision_score(y, probability)),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}


def threshold_sweep(y, probability):
    return pd.DataFrame([{"Threshold": float(t), **scores(y, probability, t)}
                         for t in np.round(np.arange(.05, .96, .01), 2)])


def choose_threshold(y, probability):
    curve = threshold_sweep(y, probability)
    best = curve.sort_values(["F2_NguyCo", "Precision_NguyCo", "Threshold"], ascending=False).iloc[0]
    return float(best.Threshold), curve


def split_dataset(profiles, records):
    required = set(FEATURES + ["MSSV", "MaHocPhan", "NguyCo"])
    if required - set(records):
        raise ValueError(f"Missing columns: {sorted(required-set(records))}")
    if records[["MSSV", "MaHocPhan", "NguyCo"]].isna().any().any():
        raise ValueError("IDs and labels cannot be missing.")
    if records.duplicated(["MSSV", "MaHocPhan"]).any() or profiles.MSSV.duplicated().any():
        raise ValueError("Duplicate student/course or profile IDs.")
    if set(records.NguyCo.unique()) != {0, 1}:
        raise ValueError("NguyCo must contain both 0 and 1.")
    if not set(records.MSSV).issubset(set(profiles.MSSV)):
        raise ValueError("Missing student profiles.")
    if np.isinf(records[NUMERIC_FEATURES].to_numpy(dtype=float)).any():
        raise ValueError("Infinite numeric values.")
    trainval, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(records, groups=records.MSSV))
    pool = records.iloc[trainval]
    fit, val = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=43).split(pool, groups=pool.MSSV))
    sets = {"train": pool.iloc[fit], "validation": pool.iloc[val], "test": records.iloc[test]}
    groups = [set(d.MSSV) for d in sets.values()]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    for name, data in sets.items():
        if data.NguyCo.nunique() != 2:
            raise ValueError(f"Both classes required in {name}.")
    return sets


def candidates():
    return {
        "Dummy Baseline": (DummyClassifier(strategy="most_frequent"), {}),
        "Logistic Regression": (LogisticRegression(max_iter=2000, class_weight="balanced"), {"model__C": [.1, 1., 10.]}),
        "Decision Tree": (DecisionTreeClassifier(class_weight="balanced", random_state=42),
                          {"model__max_depth": [3, 6], "model__min_samples_leaf": [5, 20]}),
        "Random Forest": (RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
                          {"model__max_depth": [6, None], "model__min_samples_leaf": [3, 10]}),
        "KNN": (KNeighborsClassifier(), {"model__n_neighbors": [9, 21], "model__weights": ["uniform", "distance"]}),
        "SVM": (SVC(probability=True, class_weight="balanced", random_state=42),
                {"model__C": [.1, 1.], "model__kernel": ["linear", "rbf"]}),
    }


def preprocessor():
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                              ("scale", StandardScaler())]), NUMERIC_FEATURES),
        ("category", Pipeline([("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                               ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL_FEATURES)])


def learning_diagnostics(estimator, data, folds):
    rows = []
    for fold, (fit_idx, val_idx) in enumerate(folds, 1):
        pool, held = data.iloc[fit_idx], data.iloc[val_idx]
        ids = np.random.default_rng(42 + fold).permutation(pool.MSSV.unique())
        for fraction in [.25, .5, 1.]:
            subset = pool.loc[pool.MSSV.isin(ids[:max(2, int(len(ids)*fraction))])]
            if subset.NguyCo.nunique() != 2:
                continue
            fitted = clone(estimator).fit(subset[FEATURES], subset.NguyCo)
            rows.append({"Fold": fold, "Fraction": fraction, "Students": subset.MSSV.nunique(),
                         "Train_AP": average_precision_score(subset.NguyCo, fitted.predict_proba(subset[FEATURES])[:, 1]),
                         "Validation_AP": average_precision_score(held.NguyCo, fitted.predict_proba(held[FEATURES])[:, 1])})
    return pd.DataFrame(rows)


def train(profiles, records, output_dir=None, folds_count=5, jobs=1):
    output_dir = Path(output_dir or MODEL_DIR)
    sets = split_dataset(profiles, records)
    data = sets["train"]
    cv = StratifiedGroupKFold(n_splits=folds_count, shuffle=True, random_state=42)
    folds = list(cv.split(data[FEATURES], data.NguyCo, data.MSSV))
    fold_manifest = []
    for number, (fit_idx, held_idx) in enumerate(folds, 1):
        fit, held = data.iloc[fit_idx], data.iloc[held_idx]
        assert not set(fit.MSSV) & set(held.MSSV)
        if min(fit.NguyCo.nunique(), held.NguyCo.nunique()) != 2:
            raise ValueError("Both classes required in every CV fold; reduce --folds.")
        fold_manifest.extend({"MSSV": sid, "HeldOutFold": number} for sid in held.MSSV.unique())
    pipelines, rows, searches = {}, [], []
    for name, (estimator, grid) in candidates().items():
        print(f"Training {name}: {folds_count} grouped folds", flush=True)
        pipe = Pipeline([("preprocess", preprocessor()), ("model", estimator)])
        search = GridSearchCV(pipe, grid, cv=folds, scoring={"AP": "average_precision", "F1": "f1", "Recall": "recall"},
                              refit="AP", n_jobs=jobs, error_score="raise", return_train_score=True)
        started = perf_counter()
        search.fit(data[FEATURES], data.NguyCo)
        pipelines[name] = search.best_estimator_
        i, res = search.best_index_, search.cv_results_
        row = {"Model": name, "Search_Seconds": perf_counter()-started, "Best_Params": json.dumps(search.best_params_, sort_keys=True)}
        for metric in ["AP", "F1", "Recall"]:
            for stat in ["mean", "std"]:
                row[f"CV_{metric}_{stat}"] = float(res[f"{stat}_test_{metric}"][i])
        rows.append(row)
        searches.append(pd.DataFrame(res).assign(Model=name))
    winner = max((r for r in rows if r["Model"] != "Dummy Baseline"), key=lambda r: r["CV_AP_mean"])["Model"]
    predictions, validations, curves = [], [], []
    for row in rows:
        name, val = row["Model"], sets["validation"]
        probability = pipelines[name].predict_proba(val[FEATURES])[:, 1]
        threshold, curve = choose_threshold(val.NguyCo, probability)
        row.update(Threshold=threshold, Selected=name == winner)
        row.update({"Validation_" + k: v for k, v in scores(val.NguyCo, probability, threshold).items()})
        curves.append(curve.assign(Model=name))
        validations.append(val[["MSSV", "MaHocPhan", "NguyCo"]].assign(Model=name, Probability=probability))
    # Freeze all choices before examining test; no refit on validation after threshold selection.
    for row in rows:
        name, test = row["Model"], sets["test"]
        probability = pipelines[name].predict_proba(test[FEATURES])[:, 1]
        row.update(scores(test.NguyCo, probability, row["Threshold"]))
        pred = probability >= row["Threshold"]
        errors = np.where(test.NguyCo.eq(1), np.where(pred, "TP", "FN"), np.where(pred, "FP", "TN"))
        predictions.append(test[["MSSV", "MaHocPhan", "NguyCo"] + FEATURES].assign(
            Model=name, Probability=probability, Threshold=row["Threshold"], Error=errors))
    print(f"Learning curves: {winner}", flush=True)
    learning = learning_diagnostics(pipelines[winner], data, folds)
    result = pd.DataFrame(rows).sort_values("CV_AP_mean", ascending=False).reset_index(drop=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = output_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    model_files = {}
    for name, pipeline in pipelines.items():
        filename = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(pipeline, run_dir / filename)
        model_files[name] = filename
    joblib.dump(pipelines[winner], run_dir / MODEL_FILE.name)
    result.to_csv(run_dir / "metrics.csv", index=False)
    pd.concat(searches, ignore_index=True).to_csv(run_dir / "cv_results.csv", index=False)
    pd.DataFrame(fold_manifest).to_csv(run_dir / "cv_folds.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(run_dir / "threshold_validation.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(run_dir / "test_predictions.csv", index=False)
    pd.concat(validations, ignore_index=True).to_csv(run_dir / "validation_predictions.csv", index=False)
    learning.to_csv(run_dir / "learning_curve.csv", index=False)
    pd.concat([d[["MSSV"]].drop_duplicates().assign(Split=n) for n, d in sets.items()]).to_csv(run_dir / "student_splits.csv", index=False)
    manifest = {"pipeline_version": "4.0", "run_id": run_id, "trained_at": datetime.now(timezone.utc).isoformat(),
                "selected_model": winner, "threshold": float(result.loc[result.Selected, "Threshold"].iloc[0]),
                "selection": "highest train grouped-CV AP; validation threshold maximizes F2",
                "cv_folds": folds_count, "seed": 42, "sklearn_version": sklearn.__version__,
                "cv_warning": "CV after tuning is optimistic; independent test is the final estimate.",
                "threshold_policy": "Grid .05..95 step .01; maximize F2, then precision, then threshold; validation only.",
                "model_files": model_files, "features": FEATURES, "dataset_rows": len(records), "students": len(profiles),
                "risk_rate": float(records.NguyCo.mean()), "synthetic": True, "positive_class": "DiemCuoiKy < 5 (synthetic)",
                "data_sha256": hashlib.sha256(records.to_csv(index=False).encode()).hexdigest(),
                "missing_values": {k: int(v) for k, v in records[FEATURES].isna().sum().items()},
                "splits": {n: {"rows": len(d), "students": d.MSSV.nunique(), "risk_rate": float(d.NguyCo.mean())} for n, d in sets.items()},
                "student_overlap": False, "calibrated": False,
                "limits": ["Synthetic data: no real-world accuracy claim.", "No timestamps: not temporal early-warning validation.",
                           "SVC internal probability fitting is not group-aware; no external calibration claimed.",
                           "What-if measures sensitivity, not causality."]}
    (run_dir / "metadata.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    previous = [p for p in output_dir.iterdir() if p.is_file()]
    if previous:
        backup = output_dir / "archive" / run_id
        backup.mkdir(parents=True, exist_ok=False)
        for path in previous:
            shutil.copy2(path, backup / path.name)
    for path in run_dir.iterdir():
        if path.name != "metadata.json":
            shutil.copy2(path, output_dir / path.name)
    shutil.copy2(run_dir / "metadata.json", output_dir / "metadata.json")
    return result
