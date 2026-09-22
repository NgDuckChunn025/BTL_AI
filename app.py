"""EduPredict AI: academic portal, simulation and model evaluation."""
from datetime import datetime
import json

import pandas as pd
import streamlit as st

from src.data_engine import load_assets, predict_risk, predict_risks, ROOT, MODEL_DIR, MODEL_FILE, COURSE_FILE, PROFILE_FILE, METRICS_FILE
from src.presentation import PAGES, ICONS, heading, kpi, risk_label, go_to, download_csv
from src.guide import render_guide
from src.evaluation_ui import render_evaluation
from src.new_student_ui import render_new_students

st.set_page_config(page_title="EduPredict AI · Academic Analytics", page_icon=":material/school:", layout="wide")


@st.cache_resource(show_spinner="Đang nạp dữ liệu học vụ…")
def get_assets(signature):
    return load_assets()


signature = tuple(p.stat().st_mtime_ns if p.exists() else 0 for p in [MODEL_FILE, COURSE_FILE, PROFILE_FILE, METRICS_FILE, MODEL_DIR / "metadata.json"])
try:
    profiles, courses, model, metrics = get_assets(signature)
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Không nạp được dữ liệu: {exc}")
    st.code(r".\venv\Scripts\python.exe -m src.training", language="powershell")
    st.stop()

st.session_state.setdefault("history", [])
threshold = float(metadata.get("threshold", 0.5))
best = metrics.loc[metrics.Model.eq(metadata["selected_model"])].iloc[0]
with st.sidebar:
    st.subheader(":material/school: EduPredict AI")
    st.caption("ACADEMIC ANALYTICS · DEMO")
    st.badge("Dữ liệu học vụ mô phỏng", icon=":material/science:", color="blue")
    page = st.radio("Không gian làm việc", PAGES, key="page",
                    format_func=lambda p: f":material/{ICONS[PAGES.index(p)]}: {p}")
    labels = profiles.set_index("MSSV").HoTen.to_dict()
    selected_id = st.selectbox("Hồ sơ sinh viên có sẵn", profiles.MSSV, key="student", disabled=page == PAGES[5],
                               format_func=lambda x: f"{labels[x]} · {x}")
    with st.popover("Cách đổi sáng / tối", icon=":material/contrast:", width="stretch"):
        st.write("Chọn **⋮ ở góc trên → Settings → Choose app theme → Light / Dark / Use system setting**.")
        st.caption("Đổi đồng bộ cả bảng, biểu đồ và ô nhập. Trình duyệt ghi nhớ lựa chọn.")
    with st.container(border=True):
        st.caption("MÔ HÌNH ĐANG SỬ DỤNG")
        st.write(f"**{best.Model}**")
        st.caption(f"Recall test {best.Recall_NguyCo:.1%} · Precision {best.Precision_NguyCo:.1%}")
        st.caption(f"Chọn bằng train CV · Ngưỡng {threshold:.0%} từ validation")

student = profiles.loc[profiles.MSSV.eq(selected_id)].iloc[0]
records = courses.loc[courses.MSSV.eq(selected_id)].copy()
records["Nguy cơ (%)"] = predict_risks(model, records) * 100
if page != PAGES[5]:
    with st.container(horizontal=True, horizontal_alignment="distribute"):
        st.badge("Học kỳ mô phỏng · 4 học phần", icon=":material/calendar_month:", color="blue")
        st.caption(f"{student.HoTen} · {student.MSSV}")


def table_data(frame):
    data = frame.copy()
    data["Trạng thái"] = data["Nguy cơ (%)"].map(lambda p: risk_label(p / 100, threshold)[0])
    return data.rename(columns={"HocPhan": "Học phần", "TinChi": "Tín chỉ", "DiemQuaTrinh": "Quá trình /10", "ChuyenCan": "Chuyên cần (%)"})


def portal():
    heading("Student success", f"Xin chào, {student.HoTen}", f"{student.Khoa} · {student.KhoaHoc} · Theo dõi học tập và chủ động cải thiện.")
    left, right = st.columns([2, 1])
    with left, st.container(border=True):
        st.subheader("Hồ sơ học tập", icon=":material/badge:")
        cols = st.columns(3)
        cols[0].metric("GPA tích luỹ", f"{student.GPA_TichLuy:.2f} / 4")
        cols[1].metric("Tín chỉ tích luỹ", int(student.TinChiTichLuy))
        cols[2].metric("Tín chỉ kỳ này", int(records.TinChi.sum()))
        st.caption("Thông tin tích luỹ trước kỳ hiện tại · Hồ sơ mô phỏng")
        st.button("Mở trình mô phỏng", icon=":material/tune:", type="primary", on_click=go_to, args=(PAGES[2],))
    with right, st.container(border=True):
        count = int((records["Nguy cơ (%)"] >= threshold * 100).sum())
        st.subheader("Tín hiệu kỳ học", icon=":material/monitoring:")
        st.badge(f"{count} học phần cần hỗ trợ" if count else "Chưa có học phần vượt ngưỡng", color="orange" if count else "green")
        st.metric("Nguy cơ trung bình", f"{records['Nguy cơ (%)'].mean():.1f}%")
        st.caption("Trung bình theo môn; không phải xác suất trượt cả kỳ.")
    with st.container(border=True):
        st.subheader("Chi tiết học phần", icon=":material/table_chart:")
        data = table_data(records)
        columns = ["MaHocPhan", "Học phần", "Tín chỉ", "Quá trình /10", "Chuyên cần (%)", "Nguy cơ (%)", "Trạng thái"]
        st.dataframe(data[columns], hide_index=True, width="stretch", column_config={
            "MaHocPhan": "Mã môn", "Nguy cơ (%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%")})
        download_csv("Xuất học phần", data[columns], f"hoc_phan_{selected_id}.csv")
    a, b = st.columns(2)
    worst = records.loc[records["Nguy cơ (%)"].idxmax()]
    with a, st.container(border=True):
        st.subheader("Ưu tiên trao đổi", icon=":material/lightbulb:")
        st.write(f"**{worst.HocPhan}** · Nguy cơ {worst['Nguy cơ (%)']:.1f}%")
        st.write(f"Chuyên cần {worst.ChuyenCan:.0f}% · Quá trình {worst.DiemQuaTrinh:.1f}/10 · Tự học {worst.GioTuHoc:.1f} giờ/tuần.")
        st.caption("Môn có điểm nguy cơ cao nhất trong hồ sơ; chưa xác định nguyên nhân nhân quả.")
    with b, st.container(border=True):
        st.subheader("Gợi ý hành động", icon=":material/checklist:")
        if worst.ChuyenCan < 90:
            st.write("Ưu tiên tham gia đầy đủ các buổi học tiếp theo.")
        if worst.TyLeNopDungHan < 90:
            st.write("Lập lịch hoàn thành bài tập trước hạn.")
        st.write("Mô phỏng thêm giờ tự học, sau đó trao đổi với cố vấn về kế hoạch phù hợp.")
        st.caption("Gợi ý theo quy tắc; không phải khuyến nghị đã kiểm chứng hiệu quả.")


def simulator():
    heading("Predictive studio", "Dự đoán & mô phỏng kịch bản", "Chỉnh hồ sơ, chạy phân tích và so sánh với dữ liệu gốc.")
    code = st.selectbox("Học phần", records.MaHocPhan, format_func=lambda c: records.set_index("MaHocPhan").loc[c, "HocPhan"], key="course")
    original = records.loc[records.MaHocPhan.eq(code)].iloc[0].to_dict()
    context = f"{selected_id}_{code}"
    st.session_state.setdefault("revisions", {})
    revision = st.session_state.revisions.get(context, 0)
    if st.button("Khôi phục hồ sơ gốc", icon=":material/restart_alt:", key="reset"):
        st.session_state.revisions[context] = revision + 1
        st.session_state.pop("result_" + context, None)
        st.rerun()
    left, right = st.columns([1, 1.35])
    specs = [("GPA_TichLuy", "GPA tích luỹ /4", 0., 4., .01),
             ("DiemQuaTrinh", "Điểm quá trình /10", 0., 10., .1),
             ("ChuyenCan", "Chuyên cần (%)", 0., 100., .1),
             ("TruyCapLMS", "Truy cập LMS /tuần", 0., 30., 1.),
             ("TyLeNopDungHan", "Nộp đúng hạn (%)", 0., 100., .1),
             ("GioTuHoc", "Tự học (giờ/tuần)", 0., 20., .1)]
    with left, st.form(f"scenario_{context}_{revision}", border=True):
        st.subheader("Thiết lập kịch bản", icon=":material/tune:")
        edited = dict(original)
        for feature, label, low, high, step in specs:
            edited[feature] = st.slider(label, low, high, float(original[feature]), step, key=f"{context}_{revision}_{feature}")
        submitted = st.form_submit_button("Phân tích kịch bản", type="primary", icon=":material/auto_awesome:", width="stretch")
    if submitted:
        st.session_state["result_" + context] = {"input": edited, "risk": predict_risk(model, edited), "model_signature": signature}
    saved = st.session_state.get("result_" + context)
    if saved and saved.get("model_signature") != signature:
        saved = None
    current = saved["input"] if saved else original
    base_risk = predict_risk(model, original)
    risk = saved["risk"] if saved else base_risk
    with right, st.container(border=True):
        st.subheader("Kết quả phân tích", icon=":material/query_stats:")
        label, color = risk_label(risk, threshold)
        st.badge(label, color=color)
        c1, c2 = st.columns(2)
        c1.metric("Khả năng đạt ngưỡng", f"{1-risk:.1%}", help="1 − điểm nguy cơ; ngưỡng điểm thi cuối kỳ của dataset là 5/10.")
        c2.metric("Nguy cơ", f"{risk:.1%}", delta=f"{(risk-base_risk)*100:+.1f} điểm % so với gốc", delta_color="inverse")
        st.progress(float(risk), text=f"Điểm nguy cơ mô hình · ngưỡng cảnh báo {threshold:.0%}")
        st.caption("Kịch bản đã phân tích" if saved else "Đang hiển thị hồ sơ gốc. Nhấn Phân tích kịch bản để áp dụng thay đổi.")
        st.subheader("Độ nhạy từng yếu tố", icon=":material/compare_arrows:")
        effects = []
        for feature, label, *_ in specs:
            if current[feature] != original[feature]:
                one = dict(original)
                one[feature] = current[feature]
                effects.append({"Yếu tố": label, "Gốc": original[feature], "Kịch bản": current[feature],
                                "Thay đổi nguy cơ (điểm %)": (predict_risk(model, one)-base_risk)*100})
        if effects:
            st.dataframe(pd.DataFrame(effects), hide_index=True, width="stretch")
        else:
            st.caption("Chưa có đầu vào thay đổi so với hồ sơ gốc.")
        st.caption("Mỗi dòng chỉ thay một yếu tố; không cộng thành tổng, không phải SHAP hoặc tác động nhân quả.")
        with st.container(horizontal=True):
            if st.button("Lưu kịch bản", icon=":material/bookmark_add:", disabled=not saved, key="save"):
                st.session_state.history.append({"Thời gian": datetime.now().isoformat(timespec="seconds"), "MSSV": selected_id,
                                                  "Môn": code, "Model": best.Model, "Nguy cơ (%)": round(risk*100, 2),
                                                  **{k: current[k] for k, *_ in specs}})
                st.toast("Đã lưu kịch bản trong phiên.", icon=":material/check_circle:")
            report = {"student": selected_id, "course": code, "model": best.Model, "risk": risk,
                      "synthetic": True, "features": {k: current[k] for k, *_ in specs}}
            st.download_button("Xuất kết quả", json.dumps(report, ensure_ascii=False, indent=2), file_name=f"kich_ban_{context}.json",
                               mime="application/json", icon=":material/download:")
    if st.session_state.history:
        with st.expander(f"Lịch sử kịch bản · {len(st.session_state.history)} lần lưu"):
            hist = pd.DataFrame(st.session_state.history)
            st.dataframe(hist, hide_index=True, width="stretch")
            download_csv("Tải lịch sử", hist, "lich_su_kich_ban.csv")


def overview(interventions=False):
    heading("Academic analytics", "Danh sách can thiệp" if interventions else "Tổng quan học vụ", "Thống kê trên các lượt sinh viên–học phần trong dữ liệu mô phỏng.")
    faculty = st.selectbox("Khoa", ["Tất cả"] + sorted(courses.Khoa.unique()), key="faculty")
    data = courses if faculty == "Tất cả" else courses.loc[courses.Khoa.eq(faculty)]
    data = data.copy()
    data["Nguy cơ (%)"] = predict_risks(model, data) * 100
    cols = st.columns(4)
    for col, label, value, note in zip(cols, ["Sinh viên", "Lượt học phần", "Cảnh báo", "Nguy cơ trung bình"],
                                     [str(data.MSSV.nunique()), f"{len(data):,}", str((data["Nguy cơ (%)"]>=threshold * 100).sum()), f"{data['Nguy cơ (%)'].mean():.1f}%"],
                                     ["Sinh viên duy nhất", "Một dòng / sinh viên / môn", f"Nguy cơ ≥ {threshold:.0%}", "Điểm dự báo, không phải tỷ lệ thực tế"]):
        with col: kpi(label, value, note)
    if interventions:
        minimum = st.slider("Ngưỡng lọc nguy cơ (%)", 0, 100, round(threshold * 100))
        filtered = data.loc[data["Nguy cơ (%)"] >= minimum].sort_values("Nguy cơ (%)", ascending=False)
        filtered = filtered.merge(profiles[["MSSV", "HoTen"]], on="MSSV")
        st.caption(f"{len(filtered):,} lượt học phần · {filtered.MSSV.nunique():,} sinh viên. Ngưỡng lọc chỉ thay đổi danh sách.")
        display = filtered[["MSSV", "HoTen", "HocPhan", "ChuyenCan", "DiemQuaTrinh", "Nguy cơ (%)"]]
        st.dataframe(display, hide_index=True, width="stretch", column_config={"Nguy cơ (%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%")})
        download_csv("Xuất danh sách đã lọc", display, "danh_sach_can_ho_tro.csv")
    else:
        left, right = st.columns(2)
        with left, st.container(border=True):
            st.subheader("Nguy cơ trung bình theo môn")
            st.bar_chart(data.groupby("HocPhan")["Nguy cơ (%)"].mean(), horizontal=True)
        with right, st.container(border=True):
            st.subheader("Phân bố cảnh báo")
            counts = data["Nguy cơ (%)"].map(lambda v: risk_label(v/100, threshold)[0]).value_counts()
            st.bar_chart(counts, horizontal=True)
        st.caption("Dashboard chứa cả dữ liệu train/validation/test; hiệu năng ngoài mẫu nằm ở trang Hiệu năng mô hình.")


def model_page():
    render_evaluation(metrics, metadata)


if page == PAGES[0]: portal()
elif page == PAGES[1]: overview()
elif page == PAGES[2]: simulator()
elif page == PAGES[3]: overview(True)
elif page == PAGES[4]: model_page()
elif page == PAGES[5]: render_new_students(model, courses, metadata)
else:
    heading("Learning center", "Hướng dẫn & ý nghĩa chỉ số", "Cách đọc kết quả, thay đổi giao diện và huấn luyện lại.")
    with st.container(horizontal=True):
        team_doc = ROOT / "report" / "Phan_cong_cong_viec_EduPredict_AI.docx"
        if team_doc.exists():
            st.download_button("Phân công nhóm · Word", team_doc.read_bytes(), file_name=team_doc.name,
                               mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", icon=":material/description:")
        manual = ROOT / "docs" / "HUONG_DAN.md"
        if manual.exists():
            st.download_button("Hướng dẫn huấn luyện", manual.read_bytes(), file_name=manual.name,
                               mime="text/markdown", icon=":material/menu_book:")
    render_guide()
    checklist = ROOT / "docs" / "QUY_TRINH_HUAN_LUYEN_KIEM_THU.md"
    if checklist.exists():
        st.download_button("Quy trình huấn luyện & kiểm thử", checklist.read_bytes(), file_name=checklist.name,
                           mime="text/markdown", icon=":material/fact_check:")
st.caption("EduPredict AI · Bài tập lớn Trí tuệ nhân tạo · Dữ liệu mô phỏng")
