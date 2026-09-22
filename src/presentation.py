"""Theme-aware native UI helpers and shared academic labels."""
import streamlit as st

PAGES = ["Cổng sinh viên", "Tổng quan", "Dự đoán & mô phỏng", "Danh sách can thiệp", "Hiệu năng mô hình", "Kiểm thử sinh viên mới", "Hướng dẫn & chỉ số"]
ICONS = ["school", "dashboard", "model_training", "support_agent", "analytics", "person_add", "menu_book"]
THRESHOLD = 0.5


def risk_label(value, threshold=THRESHOLD):
    if value >= threshold:
        return "Cần hỗ trợ", "red"
    if value >= threshold * 0.6:
        return "Theo dõi", "orange"
    return "Nguy cơ thấp", "green"


def heading(kicker, title, subtitle):
    st.caption(kicker.upper())
    st.title(title)
    st.caption(subtitle)


def kpi(label, value, note="", help=None):
    with st.container(border=True):
        st.metric(label, value, help=help)
        if note:
            st.caption(note)


def go_to(page):
    st.session_state["page"] = page


def download_csv(label, frame, filename, key=None):
    st.download_button(label, frame.to_csv(index=False).encode("utf-8-sig"),
                       file_name=filename, mime="text/csv", icon=":material/download:", key=key)
