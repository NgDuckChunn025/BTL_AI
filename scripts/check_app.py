"""Regression checks for the actual prediction, reset and export workflows."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from streamlit.testing.v1 import AppTest
from src.presentation import PAGES
from src.new_student import template_csv, parse_csv
from src.data_engine import COURSE_FILE
import pandas as pd

at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=45).run()
assert not at.exception, at.exception
for page in PAGES:
    at.radio(key="page").set_value(page).run()
    assert not at.exception, (page, at.exception)
at.radio(key="page").set_value(PAGES[4]).run()
assert len(at.selectbox(key="evaluation_model").options) == 6
for section in ["Kết quả test", "So sánh CV", "Khảo sát ngưỡng", "Đường cong học", "Dữ liệu & giới hạn"]:
    at.segmented_control(key="evaluation_section").set_value(section).run()
    assert not at.exception, (section, at.exception)
at.segmented_control(key="evaluation_section").set_value("Khảo sát ngưỡng").run()
for name in at.selectbox(key="evaluation_model").options:
    at.selectbox(key="evaluation_model").set_value(name).run()
    assert not at.exception, (name, at.exception)
    at.slider[0].set_value(.50).run()
    assert not at.exception
at.radio(key="page").set_value(PAGES[2]).run()
assert at.button(key="save").disabled
at.slider[1].set_value(2.0)
submit = next(b for b in at.button if b.label == "Phân tích kịch bản")
submit.click().run()
assert not at.exception, at.exception
assert not at.button(key="save").disabled
at.button(key="save").click().run()
assert len(at.session_state.history) == 1
assert at.session_state.history[0]["DiemQuaTrinh"] == 2.0
at.button(key="reset").click().run()
assert at.button(key="save").disabled
assert not at.exception
at.selectbox(key="student").set_value(at.selectbox(key="student").options[1].split(" · ")[-1]).run()
assert not at.exception
at.radio(key="page").set_value(PAGES[5]).run()
at.button(key="new_predict").click().run()
assert at.error and not at.exception, "Empty ID must fail"
at.text_input(key="new_id").set_value("NEW_CHECK_001")
at.button(key="new_predict").click().run()
assert not at.exception and not at.error, at.exception
saved = at.session_state.new_single_result
assert "NguyCo" not in saved["frame"]
assert saved["frame"].Probability.between(0, 1).all()
at.button(key="new_compare").click().run()
assert at.error, "Missing actual grade must not count as zero"
actual = next(n for n in at.number_input if n.key.startswith("new_actual_"))
actual.set_value(4.0)
at.button(key="new_compare").click().run()
assert not at.exception and at.session_state.new_single_result["actual"] == 4.0
# Re-submission creates a fresh result, not the previous ground truth.
at.number_input(key="new_DiemQuaTrinh").set_value(3.0)
at.button(key="new_predict").click().run()
assert not at.exception and "actual" not in at.session_state.new_single_result
at.text_input(key="new_id").set_value(pd.read_csv(COURSE_FILE).MSSV.iloc[0])
at.button(key="new_predict").click().run()
assert at.error and "new_single_result" not in at.session_state
at.segmented_control(key="new_mode").set_value("Lô CSV").run()
at.segmented_control(key="new_csv_method").set_value("Dán CSV").run()
reference = pd.read_csv(COURSE_FILE)
sample = parse_csv(template_csv(reference, True))
second = sample.assign(MSSV="NEW_CHECK_002", DiemCuoiKy="8", DiemQuaTrinh="8.0")
batch = pd.concat([sample, second], ignore_index=True)
at.text_area(key="new_csv_text").set_value(batch.to_csv(index=False))
at.button(key="new_batch_run").click().run()
assert at.error, "Require source acknowledgement"
at.checkbox(key="new_batch_ack").check()
at.button(key="new_batch_run").click().run()
assert not at.exception and not at.error, at.exception
assert len(at.session_state.new_batch_result["frame"]) == 2
assert any(m.label == "Accuracy" for m in at.metric)
# A changed payload must not display stale metrics before resubmission.
at.text_area(key="new_csv_text").set_value("bad,csv\n1,2").run()
assert not any(m.label == "Accuracy" for m in at.metric)
at.button(key="new_batch_run").click().run()
assert at.error and "new_batch_result" not in at.session_state
# Unlabelled prediction must not invent accuracy.
at.text_area(key="new_csv_text").set_value(template_csv(reference).decode("utf-8-sig"))
at.button(key="new_batch_run").click().run()
assert not at.exception and not at.error
assert not any(m.label == "Accuracy" for m in at.metric)
print("PASS: seven pages, evaluation, scenarios, new-student input/ground truth, CSV validation/metrics, stale-result guards")
