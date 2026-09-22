"""Unit/integration tests for new-student parsing, validation, prediction and metrics."""
from pathlib import Path
import hashlib
import json
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from src.data_engine import COURSE_FILE, PROFILE_FILE, FEATURES, MODEL_DIR, MODEL_FILE
from src.new_student import (parse_csv, validate_new_students, predict_new_students, evaluate_new_students,
                             template_csv, safe_csv, MAX_ROWS)


class NewStudentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = pd.read_csv(COURSE_FILE)
        cls.meta = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
        cls.model = joblib.load(MODEL_FILE)
        cls.paths = [COURSE_FILE, PROFILE_FILE, MODEL_FILE, MODEL_DIR / "metrics.csv", MODEL_DIR / "metadata.json"]
        cls.before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in cls.paths}

    @classmethod
    def tearDownClass(cls):
        assert cls.before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in cls.paths}, "Testing mutated assets"

    def sample(self, labelled=False):
        return parse_csv(template_csv(self.reference, labelled))

    def validate(self, sample):
        return validate_new_students(sample, self.reference)

    def test_unlabelled(self):
        clean, errors, _, independent = self.validate(self.sample())
        self.assertFalse(errors)
        self.assertTrue(independent)
        result = predict_new_students(self.model, clean, self.meta)
        self.assertTrue(result.Probability.between(0, 1).all())
        self.assertIsNone(evaluate_new_students(result))

    def test_target_never_used_as_feature(self):
        clean, errors, *_ = self.validate(self.sample(True))
        self.assertFalse(errors)
        first = predict_new_students(self.model, clean, self.meta)
        second = predict_new_students(self.model, clean.assign(NguyCo=0, DiemCuoiKy=9), self.meta)
        np.testing.assert_array_equal(first.Probability, second.Probability)
        self.assertEqual(list(self.model.feature_names_in_), FEATURES)

    def test_bad_numeric_and_missing_values(self):
        for column, value in [("GPA_TichLuy", "5"), ("DiemQuaTrinh", "-1"), ("ChuyenCan", "101"),
                              ("TruyCapLMS", "2.5"), ("GioTuHoc", "inf"), ("TinChiTichLuy", ""), ("GPA_TichLuy", "abc")]:
            with self.subTest(column=column, value=value):
                sample = self.sample()
                sample.loc[0, column] = value
                self.assertTrue(self.validate(sample)[1])

    def test_missing_column(self):
        self.assertTrue(self.validate(self.sample().drop(columns="ChuyenCan"))[1])

    def test_existing_student_case_insensitive(self):
        sample = self.sample()
        sample.loc[0, "MSSV"] = " " + self.reference.MSSV.iloc[0].lower() + " "
        self.assertTrue(self.validate(sample)[1])

    def test_invalid_id_and_duplicates(self):
        self.assertTrue(self.validate(self.sample().assign(MSSV="=IMPORTXML(1)"))[1])
        self.assertTrue(self.validate(pd.concat([self.sample(), self.sample()]))[1])

    def test_renamed_training_row_not_independent(self):
        sample = self.reference.iloc[[0]].assign(MSSV="NEW_RENAMED")
        _, errors, warnings, independent = self.validate(sample)
        self.assertFalse(errors)
        self.assertFalse(independent)
        self.assertTrue(warnings)

    def test_unseen_category_and_outside_observed(self):
        sample = self.sample().assign(Khoa="Khoa mới", GioTuHoc="100")
        clean, errors, warnings, _ = self.validate(sample)
        self.assertFalse(errors)
        self.assertGreaterEqual(len(warnings), 2)
        self.assertTrue(predict_new_students(self.model, clean, self.meta).Probability.notna().all())

    def test_course_mismatch(self):
        self.assertTrue(self.validate(self.sample().assign(TinChi="10"))[1])

    def test_label_consistency(self):
        for sample in [self.sample(True).assign(NguyCo="0"), self.sample().assign(NguyCo="2"),
                       self.sample(True).assign(DiemCuoiKy=""), self.sample(True).assign(DiemCuoiKy="11")]:
            self.assertTrue(self.validate(sample)[1])

    def test_boundary_grade_five_is_not_risk(self):
        clean, errors, *_ = self.validate(self.sample(True).assign(DiemCuoiKy="5"))
        self.assertFalse(errors)
        self.assertEqual(clean.NguyCo.iloc[0], 0)

    def test_single_class_metrics(self):
        result = pd.DataFrame({"MSSV": ["A", "B"], "NguyCo": [0, 0], "Prediction": [0, 0], "Probability": [.01, .02]})
        metrics = evaluate_new_students(result)
        self.assertEqual(metrics["Accuracy"], 1)
        for key in ["Recall", "Precision", "F1", "ROC_AUC", "AP"]:
            self.assertIsNone(metrics[key])

    def test_metrics_known_confusion(self):
        result = pd.DataFrame({"MSSV": list("ABCD"), "NguyCo": [1, 1, 0, 0], "Prediction": [1, 0, 1, 0], "Probability": [.9, .2, .8, .1]})
        metrics = evaluate_new_students(result)
        self.assertEqual([metrics[k] for k in ["TP", "FN", "FP", "TN"]], [1, 1, 1, 1])
        self.assertEqual(metrics["Accuracy"], .5)
        self.assertEqual(metrics["Recall"], .5)

    def test_csv_bom_separator_and_leading_zero(self):
        csv_data = self.sample().assign(MSSV="0000007").to_csv(index=False, sep=";").encode("utf-8-sig")
        self.assertEqual(parse_csv(csv_data).MSSV.iloc[0], "0000007")

    def test_malformed_csv(self):
        for payload in [b"", b"A,A\n1,2", b"A,B\n1,2,3", b"\xff\xfe", b"A\n\"unterminated"]:
            with self.assertRaises(ValueError):
                parse_csv(payload)

    def test_limits(self):
        with self.assertRaises(ValueError):
            parse_csv(b"a" * (2 * 1024 * 1024 + 1))
        self.assertTrue(self.validate(pd.concat([self.sample()] * (MAX_ROWS+1)))[1])

    def test_safe_export(self):
        exported = safe_csv(pd.DataFrame({"Text": ["=1+1", "+CMD", " @X", "safe"], "Score": [.3, .2, .4, .1]})).decode("utf-8-sig")
        self.assertIn("'=1+1", exported)
        self.assertIn("'+CMD", exported)


if __name__ == "__main__":
    unittest.main(verbosity=2)
