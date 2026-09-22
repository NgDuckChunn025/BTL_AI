"""New-student prediction sandbox and labelled external evaluation."""
import json
from datetime import datetime, timezone
import hashlib

import pandas as pd
import streamlit as st
from src.new_student import (NUMERIC_INPUTS, parse_csv, validate_new_students, predict_new_students,
                             evaluate_new_students, template_csv, input_digest, safe_csv)
from src.presentation import heading, risk_label

ORIGINS = ["Nhập thử / dữ liệu mô phỏng", "Kết quả thực tế đã ẩn danh"]


def show_validation(errors, warnings):
    for error in errors:
        st.error(error)
    if warnings:
        with st.expander(f"Lưu ý về dữ liệu · {len(warnings)}", expanded=True):
            for warning in warnings:
                st.warning(warning)


def export_result(result, metadata, origin, independent, key):
    summary = evaluate_new_students(result) if independent else None
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "model": metadata["selected_model"],
              "run_id": metadata["run_id"], "threshold": metadata["threshold"], "training_synthetic": True,
              "source_declared_by_user": origin, "no_known_id_or_exact_feature_overlap": independent,
              "input_sha256": input_digest(result), "metrics": summary,
              "warning": "No overlap detected is not proof of independence. Manual examples are not real-world validation.",
              "records": result.to_dict(orient="records")}
    with st.container(horizontal=True):
        st.download_button("Tải kết quả CSV", safe_csv(result), file_name="kiem_thu_sinh_vien_moi.csv", mime="text/csv",
                           icon=":material/download:", key=key + "_csv")
        st.download_button("Tải báo cáo JSON", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                           file_name="bao_cao_kiem_thu.json", mime="application/json", icon=":material/description:", key=key + "_json")


def render_single(model, reference, metadata):
    left, right = st.columns([1.25, 1])
    with left:
        catalog = reference.drop_duplicates("MaHocPhan").set_index("MaHocPhan")
        course = st.selectbox("Học phần cần dự đoán", list(catalog.index),
                              format_func=lambda c: f"{c} · {catalog.loc[c, 'HocPhan']}", key="new_course")
        with st.form("new_student_form", border=True):
            st.subheader("Hồ sơ chưa có trong dataset", icon=":material/person_add:")
            student_id = st.text_input("Mã sinh viên mới / mã ẩn danh", placeholder="Ví dụ: NEW_2026001", max_chars=40, key="new_id")
            origin = st.selectbox("Nguồn hồ sơ", ORIGINS, key="new_origin")
            a, b = st.columns(2)
            faculty = a.selectbox("Khoa", sorted(reference.Khoa.unique()), key="new_faculty")
            group = b.selectbox("Nhóm học", sorted(reference.NhomHoc.unique()), key="new_group")
            values = {}
            st.caption("Thông tin tích luỹ & học phần · chỉ nhập dữ liệu có trước khi thi")
            fields = [f for f in NUMERIC_INPUTS if f != "TinChi"]
            cols = st.columns(2)
            for i, field in enumerate(fields):
                label, low, high, default, step = NUMERIC_INPUTS[field]
                values[field] = cols[i % 2].number_input(label, min_value=low, max_value=high, value=default, step=step, key="new_" + field)
            st.caption(f"Tín chỉ học phần: {int(catalog.loc[course, 'TinChi'])}. Không nhập họ tên hoặc thông tin nhạy cảm.")
            submitted = st.form_submit_button("Dự đoán sinh viên mới", type="primary", icon=":material/play_arrow:", width="stretch", key="new_predict")
        if submitted:
            st.session_state.pop("new_single_result", None)
            record = {"MSSV": student_id, "MaHocPhan": course, "HocPhan": catalog.loc[course, "HocPhan"],
                      "TinChi": int(catalog.loc[course, "TinChi"]), "Khoa": faculty, "NhomHoc": group, **values}
            clean, errors, warnings, independent = validate_new_students(pd.DataFrame([record]), reference)
            if errors:
                show_validation(errors, warnings)
            else:
                predicted = predict_new_students(model, clean, metadata)
                st.session_state.new_single_result = {"frame": predicted, "origin": origin, "warnings": warnings,
                                                      "independent": independent, "run_id": metadata["run_id"],
                                                      "digest": input_digest(clean)}
    with right, st.container(border=True):
        st.subheader("Kết quả của hồ sơ đã gửi", icon=":material/query_stats:")
        saved = st.session_state.get("new_single_result")
        if not saved:
            st.info("Nhập hồ sơ và nhấn Dự đoán. Chưa có kết quả thực tế thì chưa thể biết dự đoán đúng hay sai.")
            return
        if saved["run_id"] != metadata["run_id"]:
            st.warning("Mô hình đã được huấn luyện lại. Hãy gửi hồ sơ lại để tránh dùng kết quả cũ.")
            return
        result = saved["frame"]
        row = result.iloc[0]
        st.write(f"**{row.MSSV} · {row.MaHocPhan}**")
        st.caption(f"{saved['origin']} · {metadata['selected_model']} · Ngưỡng cố định {metadata['threshold']:.0%}")
        label, color = risk_label(row.Probability, metadata["threshold"])
        st.badge(label, color=color)
        st.metric("Điểm nguy cơ dự báo", f"{row.Probability:.1%}")
        st.caption("Không phải độ chính xác của hồ sơ. Kết quả chỉ đổi khi nhấn Dự đoán, không theo các ô chưa gửi.")
        show_validation([], saved["warnings"])
        with st.expander("Kiểm tra đầu vào đã dùng"):
            st.dataframe(result.drop(columns=["Probability", "Prediction", "Threshold", "Model", "RunID"]).astype(str).T, width="stretch")
        st.subheader("Đối chiếu sau khi có kết quả", icon=":material/fact_check:")
        token = saved["digest"] + saved["run_id"]
        with st.form("new_truth_" + token, border=False):
            grade = st.number_input("Điểm thi cuối kỳ để đối chiếu /10", min_value=0., max_value=10., value=None, step=.1,
                                    placeholder="Chỉ nhập khi có kết quả", key="new_actual_" + token)
            checked = st.form_submit_button("Đối chiếu đúng / sai", icon=":material/check_circle:", key="new_compare")
        if checked:
            if grade is None:
                st.error("Chưa có điểm thực tế để đối chiếu.")
            else:
                # Compare to the frozen prediction, never rerun with the outcome as a feature.
                truth_input = result.drop(columns=["Probability", "Prediction", "Threshold", "Model", "RunID"]).copy()
                truth_input["DiemCuoiKy"] = grade
                _clean, errors, _warnings, _independent = validate_new_students(truth_input, reference)
                if errors:
                    show_validation(errors, [])
                else:
                    saved["actual"] = float(grade)
        if "actual" in saved:
            actual = int(saved["actual"] < 5)
            correct = int(row.Prediction) == actual
            (st.success if correct else st.warning)("Dự đoán đúng cho hồ sơ này." if correct else "Dự đoán sai cho hồ sơ này.")
            st.caption(f"Điểm đối chiếu {saved['actual']:.1f}/10 → nhãn {actual}. Một hồ sơ không đủ kết luận độ chính xác chung.")
            result = result.assign(DiemCuoiKy=saved["actual"], NguyCo=actual, Correct=correct)
        else:
            st.info("Chưa đối chiếu: chưa có kết quả cuối kỳ. Không hiển thị Accuracy cho một dự đoán chưa có nhãn.")
        export_result(result, metadata, saved["origin"], saved["independent"], "new_single_export")


def render_batch(model, reference, metadata):
    with st.container(border=True):
        st.subheader("Kiểm thử nhiều hồ sơ", icon=":material/upload_file:")
        st.caption("Tối đa 10.000 dòng / 2 MB. Cột nhãn là tuỳ chọn; nếu có phải đầy đủ. Dùng mã ẩn danh, không cần họ tên.")
        with st.container(horizontal=True):
            st.download_button("CSV mẫu chưa có nhãn", template_csv(reference), file_name="mau_nhap_thu.csv", mime="text/csv", icon=":material/download:")
            st.download_button("CSV mẫu có điểm đối chiếu", template_csv(reference, True), file_name="mau_co_nhan_gia_lap.csv", mime="text/csv", icon=":material/download:")
        st.warning("Dòng trong file mẫu là ví dụ tự đặt, kể cả điểm cuối kỳ; chỉ kiểm tra chức năng, không dùng chứng minh độ chính xác thực tế.")
        method = st.segmented_control("Cách nhập CSV", ["Tải file", "Dán CSV"], default="Tải file", key="new_csv_method")
        if method == "Dán CSV":
            text = st.text_area("Nội dung CSV UTF-8", height=150, key="new_csv_text")
            payload = text.encode("utf-8")
        else:
            uploaded = st.file_uploader("Chọn file CSV", type=["csv"], max_upload_size=2, key="new_csv_upload")
            payload = uploaded.getvalue() if uploaded is not None else b""
        origin = st.selectbox("Nguồn lô kiểm thử", ORIGINS, key="new_batch_origin")
        acknowledgement = st.checkbox("Nhãn (nếu có) lấy từ kết quả độc lập, không sửa theo dự đoán; dữ liệu thật đã được ẩn danh và có quyền sử dụng.", key="new_batch_ack")
        token = hashlib.sha256(payload + origin.encode() + metadata["run_id"].encode()).hexdigest()
        if st.button("Kiểm tra dữ liệu & chạy", type="primary", icon=":material/play_arrow:", key="new_batch_run"):
            st.session_state.pop("new_batch_result", None)
            if not acknowledgement:
                st.error("Hãy xác nhận nguồn dữ liệu và nhãn trước khi chạy.")
            elif not payload.strip():
                st.error("Chưa có nội dung CSV.")
            else:
                try:
                    clean, errors, warnings, independent = validate_new_students(parse_csv(payload), reference)
                    show_validation(errors, [])
                    if not errors:
                        with st.spinner("Đang dự đoán bằng mô hình đã cố định…"):
                            result = predict_new_students(model, clean, metadata)
                        st.session_state.new_batch_result = {"token": token, "frame": result, "warnings": warnings, "independent": independent}
                except (ValueError, TypeError) as exc:
                    st.error(f"Không thể xử lý CSV: {exc}")
    saved = st.session_state.get("new_batch_result")
    if not saved:
        return
    if saved["token"] != token or not acknowledgement:
        st.info("Nội dung / nguồn / mô hình đã đổi. Nhấn Kiểm tra dữ liệu & chạy để cập nhật kết quả.")
        return
    result = saved["frame"]
    show_validation([], saved["warnings"])
    with st.container(horizontal=True):
        st.metric("Sinh viên mới", result.MSSV.nunique(), border=True)
        st.metric("Lượt học phần", len(result), border=True)
        st.metric("Cảnh báo", int(result.Prediction.sum()), border=True)
        st.metric("Có nhãn đối chiếu", len(result) if "NguyCo" in result else 0, border=True)
    metrics = evaluate_new_students(result) if saved["independent"] else None
    if metrics is None:
        st.info("Chỉ có dự đoán. Để đánh giá cần nhãn NguyCo (0/1) hoặc DiemCuoiKy (0–10), và không trùng hồ sơ bộ dữ liệu hiện tại.")
    else:
        st.subheader("Đánh giá trên lô vừa cung cấp")
        if origin == ORIGINS[0]:
            st.warning("Chỉ số trên dữ liệu nhập thử / mô phỏng; không chứng minh khả năng dự báo sinh viên thật.")
        st.caption("Nguồn do người dùng khai báo, chưa được xác minh. Không trùng MSSV/đặc trưng không tự chứng minh tính độc lập hoặc tính đại diện.")
        with st.container(horizontal=True):
            for field in ["Accuracy", "Precision", "Recall", "F1"]:
                value = metrics[field]
                st.metric(field, "Không xác định" if value is None else f"{value:.1%}", border=True)
        st.table(pd.DataFrame([[metrics["TP"], metrics["FN"]], [metrics["FP"], metrics["TN"]]],
                             index=["Thực tế nguy cơ", "Thực tế không nguy cơ"], columns=["Dự báo nguy cơ", "Dự báo không nguy cơ"]))
        st.write(f"Phát hiện {metrics['TP']} / {metrics['Positive']} ca nguy cơ; bỏ sót {metrics['FN']}; báo nhầm {metrics['FP']}.")
        if metrics["ROC_AUC"] is None:
            st.warning("Lô chỉ có một lớp: không báo ROC-AUC/AP để tránh diễn giải sai. Recall không xác định nếu không có ca nguy cơ.")
        else:
            st.caption(f"ROC-AUC {metrics['ROC_AUC']:.3f} · Average Precision {metrics['AP']:.3f}")
        st.caption("Chỉ số tính theo lượt học phần, không theo người. Nhiều môn cùng sinh viên không phải những quan sát độc lập.")
        st.warning("Không kết luận hệ thống đạt chuẩn từ vài hồ sơ tự chọn. Không chỉnh mô hình/ngưỡng theo lô này rồi báo lại nó như tập kiểm thử mới.")
    display_columns = ["MSSV", "MaHocPhan", "Probability", "Threshold", "Prediction"] + [c for c in ["DiemCuoiKy", "NguyCo", "Correct", "Error"] if c in result]
    st.dataframe(result[display_columns], hide_index=True, width="stretch")
    export_result(result, metadata, origin, saved["independent"], "new_batch_export")


def render_new_students(model, reference, metadata):
    heading("New student lab", "Kiểm thử sinh viên mới", "Nhập hồ sơ → dự đoán cố định → đối chiếu khi có kết quả. Không huấn luyện lại từ dữ liệu nhập thử.")
    st.caption(f"Model: {metadata['selected_model']} · Ngưỡng {metadata['threshold']:.0%} · Run {metadata['run_id']}")
    st.info("Mô hình học từ dữ liệu mô phỏng. Điểm nguy cơ không phải độ chính xác. Kết quả thử nghiệm chỉ dùng tham khảo, không ra quyết định học vụ.")
    mode = st.segmented_control("Chế độ kiểm thử", ["Một sinh viên", "Lô CSV"], default="Một sinh viên", key="new_mode")
    if mode == "Lô CSV":
        render_batch(model, reference, metadata)
    else:
        render_single(model, reference, metadata)
    st.caption("Dữ liệu nhập chỉ xử lý trong phiên; không ghi vào CSV huấn luyện, không cập nhật benchmark. Tải báo cáo nếu cần lưu.")
