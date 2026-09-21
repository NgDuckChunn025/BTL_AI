"""
============================================================================
EDUPREDICT AI — ACADEMIC ANALYTICS
============================================================================
Ứng dụng Streamlit hỗ trợ dự đoán khả năng Đỗ/Trượt của sinh viên,
được xây dựng cho BTL môn Trí tuệ nhân tạo.

Cấu trúc ứng dụng gồm 6 trang:
    1. Giới thiệu           - Mô tả bài toán theo khung Task / Performance / Experience
    2. Tổng quan             - Dashboard thống kê tổng hợp trên toàn bộ dữ liệu
    3. Phương pháp luận      - Đối chiếu trực tiếp với "Nguyên tắc bắt buộc" và
                               "Sai lầm thường gặp" đã học trong Chương 4
    4. Dự đoán cá nhân       - Nhập thông tin sinh viên, xem dự đoán, lưu lịch sử
    5. Khám phá dữ liệu      - Bảng dữ liệu có thể lọc, biểu đồ trực quan hóa
    6. Hiệu năng mô hình     - So sánh 5 mô hình + Dummy Baseline, Cross-Validation

Toàn bộ dữ liệu hiển thị đều lấy từ pipeline thật (notebooks 01-04), không có
số liệu giả định trong phần thống kê/mô hình.
============================================================================
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
from html import escape


# Resolve every project asset relative to this file. This keeps the app working
# whether it is launched from the repository root or another working directory.
BASE_DIR = Path(__file__).resolve().parent

# ============================================================================
# 1. CẤU HÌNH TRANG
# ============================================================================
st.set_page_config(
    page_title="EduPredict AI — Academic Analytics",
    page_icon=str(BASE_DIR / "assets" / "edupredict_ai_logo.png")
    if (BASE_DIR / "assets" / "edupredict_ai_logo.png").exists() else "🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# 2. BẢNG MÀU & DESIGN TOKENS
# ============================================================================
BG_PAGE = "#0F1224"
CARD_BG = "#1B2038"
CARD_BG_HOVER = "#212748"
BORDER = "#2B3155"
BLUE = "#4C6FFF"
BLUE_SOFT = "rgba(76, 111, 255, 0.12)"
INDIGO = "#7C6CFF"
GREEN = "#22C55E"
GREEN_SOFT = "rgba(34, 197, 94, 0.10)"
RED = "#EF4444"
RED_SOFT = "rgba(239, 68, 68, 0.10)"
AMBER = "#F59E0B"
AMBER_SOFT = "rgba(245, 158, 11, 0.10)"
TEXT_MAIN = "#F3F4F6"
TEXT_MUTED = "#8891A5"


def build_css() -> str:
    """Toàn bộ CSS tùy chỉnh cho giao diện — theme tối, thẻ bo góc, hiệu ứng hover nhẹ."""
    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
        .block-container {{ padding-top: 1.1rem; padding-bottom: 3rem; max-width: 1300px; }}

        /* ---------------- Thẻ KPI ---------------- */
        .kpi-card {{
            background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 16px;
            padding: 18px 20px; height: 100%; transition: all 0.18s ease;
        }}
        .kpi-card:hover {{ background: {CARD_BG_HOVER}; border-color: {BLUE}; }}
        .kpi-label {{
            color: {TEXT_MUTED}; font-size: 0.78rem; margin-bottom: 6px;
            text-transform: uppercase; letter-spacing: 0.04em; font-weight: 600;
        }}
        .kpi-value {{ font-size: 2rem; font-weight: 800; color: {TEXT_MAIN}; line-height: 1.15; }}
        .kpi-sub {{ color: {TEXT_MUTED}; font-size: 0.78rem; margin-top: 4px; }}

        /* ---------------- Pill / badge ---------------- */
        .pill {{
            display: inline-block; background: {CARD_BG}; border: 1px solid {BORDER};
            border-radius: 999px; padding: 5px 14px; font-size: 0.8rem;
            color: {TEXT_MUTED}; margin-right: 8px;
        }}
        .status-dot {{
            height: 8px; width: 8px; border-radius: 50%;
            background: {GREEN}; display: inline-block; margin-right: 6px;
            box-shadow: 0 0 0 3px {GREEN_SOFT};
        }}

        /* ---------------- Tiêu đề ---------------- */
        .page-title {{
            font-size: 1.85rem; font-weight: 800; margin-bottom: 0;
            background: linear-gradient(90deg, {TEXT_MAIN}, {BLUE});
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .page-sub {{ color: {TEXT_MUTED}; font-size: 0.95rem; margin-top: 2px; }}
        .section-title {{ font-size: 1.02rem; font-weight: 700; color: {TEXT_MAIN}; margin: 6px 0 12px; }}
        .section-caption {{ color: {TEXT_MUTED}; font-size: 0.82rem; margin-bottom: 14px; }}

        /* ---------------- Card viền trái màu (checklist) ---------------- */
        .rule-card {{
            background: {CARD_BG}; border: 1px solid {BORDER}; border-left: 4px solid {GREEN};
            border-radius: 10px; padding: 14px 16px; margin-bottom: 10px;
        }}
        .rule-card.warn {{ border-left-color: {AMBER}; }}
        .rule-card.bad {{ border-left-color: {RED}; }}
        .rule-title {{ font-weight: 700; color: {TEXT_MAIN}; font-size: 0.92rem; margin-bottom: 4px; }}
        .rule-body {{ color: {TEXT_MUTED}; font-size: 0.85rem; line-height: 1.5; }}
        .rule-status {{
            display: inline-block; font-size: 0.72rem; font-weight: 700; padding: 2px 9px;
            border-radius: 999px; margin-bottom: 6px; letter-spacing: 0.03em;
        }}
        .rule-status.ok {{ background: {GREEN_SOFT}; color: {GREEN}; }}
        .rule-status.warn {{ background: {AMBER_SOFT}; color: {AMBER}; }}
        .rule-status.bad {{ background: {RED_SOFT}; color: {RED}; }}

        /* ---------------- Kết quả dự đoán ---------------- */
        .result-pass {{
            background: {GREEN_SOFT}; border: 1px solid {GREEN};
            border-radius: 16px; padding: 18px; text-align: center;
        }}
        .result-fail {{
            background: {RED_SOFT}; border: 1px solid {RED};
            border-radius: 16px; padding: 18px; text-align: center;
        }}

        /* ---------------- Ma trận nhầm lẫn ---------------- */
        .cm-box {{ border-radius: 14px; padding: 18px; text-align: center; border: 1px solid {BORDER}; }}
        .cm-label {{ font-size: 0.74rem; opacity: 0.9; letter-spacing: 0.02em; }}
        .cm-value {{ font-size: 1.8rem; font-weight: 800; }}

        /* ---------------- Thanh phân bố HTML ---------------- */
        .bar-row {{ margin-bottom: 14px; }}
        .bar-row-head {{ display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 4px; }}
        .bar-row-label {{ color: #E5E7EB; }}
        .bar-row-pct {{ color: {TEXT_MUTED}; }}
        .bar-track {{ background: rgba(255,255,255,0.06); border-radius: 5px; height: 8px; overflow: hidden; }}
        .bar-fill {{ height: 100%; border-radius: 5px; transition: width 0.4s ease; }}

        /* ---------------- Gauge tròn ---------------- */
        .gauge-wrap {{ display: flex; align-items: center; justify-content: center; margin: 10px 0 18px; }}
        .gauge-inner {{
            width: 104px; height: 104px; border-radius: 50%; background: {BG_PAGE};
            display: flex; flex-direction: column; align-items: center; justify-content: center;
        }}
        .gauge-value {{ font-size: 1.3rem; font-weight: 800; color: {TEXT_MAIN}; }}
        .gauge-caption {{ font-size: 0.68rem; color: {TEXT_MUTED}; }}

        /* ---------------- Bảng HTML thuần ---------------- */
        .styled-table {{ width: 100%; font-size: 0.85rem; border-collapse: collapse; }}
        .styled-table th {{
            text-align: left; padding: 9px 8px; color: {TEXT_MUTED}; font-weight: 600;
            border-bottom: 1px solid {BORDER}; font-size: 0.76rem; text-transform: uppercase;
            letter-spacing: 0.03em;
        }}
        .styled-table td {{ padding: 10px 8px; border-bottom: 1px solid {BORDER}; color: #E5E7EB; }}
        .styled-table tr:last-child td {{ border-bottom: none; }}
        .styled-table tr:hover td {{ background: rgba(255,255,255,0.02); }}

        /* ---------------- Timeline / step ---------------- */
        .step-item {{ display: flex; gap: 12px; margin-bottom: 16px; }}
        .step-num {{
            flex-shrink: 0; width: 28px; height: 28px; border-radius: 50%;
            background: {BLUE_SOFT}; color: {BLUE}; display: flex; align-items: center;
            justify-content: center; font-weight: 700; font-size: 0.85rem;
        }}
        .step-title {{ font-weight: 700; color: {TEXT_MAIN}; font-size: 0.9rem; }}
        .step-body {{ color: {TEXT_MUTED}; font-size: 0.83rem; margin-top: 2px; line-height: 1.5; }}

        /* ---------------- Sidebar ---------------- */
        section[data-testid="stSidebar"] {{ border-right: 1px solid {BORDER}; }}
        .sidebar-brand-name {{ font-weight: 800; font-size: 1.15rem; color: {TEXT_MAIN}; margin-bottom: 0; }}
        .sidebar-brand-sub {{ font-size: 0.7rem; color: {TEXT_MUTED}; letter-spacing: 0.08em; }}

        div[data-testid="stMetric"] {{
            background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 12px; padding: 12px 16px;
        }}
    </style>
    """


st.markdown(build_css(), unsafe_allow_html=True)

# ============================================================================
# 3. HÀM HỖ TRỢ DỰNG GIAO DIỆN (UI HELPERS)
# ============================================================================


def kpi_card(label: str, value: str, sub: str = "") -> None:
    """Vẽ 1 thẻ KPI (số liệu tổng hợp) theo đúng phong cách dashboard."""
    st.markdown(
        f"""<div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def cm_box(label: str, value, bg: str, border: str) -> None:
    """Vẽ 1 ô trong ma trận nhầm lẫn (Confusion Matrix)."""
    st.markdown(
        f"""<div class="cm-box" style="background:{bg}; border-color:{border};">
                <div class="cm-label">{label}</div>
                <div class="cm-value" style="color:{border};">{value}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def html_bar(label: str, pct: float, color: str) -> None:
    """Thanh phân bố dạng HTML/CSS thuần (không dùng thư viện chart ngoài)."""
    pct_clamped = max(0.0, min(pct, 1.0)) * 100
    st.markdown(
        f"""<div class="bar-row">
                <div class="bar-row-head">
                    <span class="bar-row-label">{label}</span>
                    <span class="bar-row-pct">{pct * 100:.0f}%</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill" style="width:{pct_clamped}%; background:{color};"></div>
                </div>
            </div>""",
        unsafe_allow_html=True,
    )


def html_gauge(value_pct: float, color: str, caption: str = "xác suất đỗ") -> None:
    """Vòng tròn gauge dùng conic-gradient CSS."""
    v = max(0.0, min(value_pct, 100.0))
    st.markdown(
        f"""<div class="gauge-wrap">
                <div style="width:140px;height:140px;border-radius:50%;
                            background:conic-gradient({color} 0% {v}%, {CARD_BG} {v}% 100%);
                            display:flex;align-items:center;justify-content:center;">
                    <div class="gauge-inner">
                        <div class="gauge-value">{v:.0f}%</div>
                        <div class="gauge-caption">{caption}</div>
                    </div>
                </div>
            </div>""",
        unsafe_allow_html=True,
    )


def html_table(df: pd.DataFrame, pct_cols=None) -> None:
    """Render một DataFrame nhỏ thành bảng HTML thuần, có thể định dạng % cho một số cột."""
    pct_cols = pct_cols or []
    header_html = "".join(f"<th>{c}</th>" for c in df.columns)
    rows_html = ""
    for _, row in df.iterrows():
        cells = ""
        for c in df.columns:
            val = row[c]
            if c in pct_cols and isinstance(val, (int, float)):
                cells += f"<td>{val:.1%}</td>"
            elif isinstance(val, float):
                cells += f"<td>{val:.3f}</td>"
            else:
                cells += f"<td>{val}</td>"
        rows_html += f"<tr>{cells}</tr>"
    st.markdown(
        f'<table class="styled-table"><thead><tr>{header_html}</tr></thead>'
        f'<tbody>{rows_html}</tbody></table>',
        unsafe_allow_html=True,
    )


def section_title(text: str, caption: str = "") -> None:
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="section-caption">{caption}</div>', unsafe_allow_html=True)


def rule_card(status: str, title: str, body: str) -> None:
    """
    Thẻ đối chiếu 1 nguyên tắc/sai lầm với thực tế project.
    status: "ok" | "warn" | "bad"
    """
    status_label = {"ok": "ĐÃ TUÂN THỦ", "warn": "CẦN LƯU Ý", "bad": "ĐÃ PHÁT HIỆN & SỬA"}
    css_class = {"ok": "", "warn": "warn", "bad": "bad"}
    st.markdown(
        f"""<div class="rule-card {css_class[status]}">
                <span class="rule-status {status}">{status_label[status]}</span>
                <div class="rule-title">{title}</div>
                <div class="rule-body">{body}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def step_item(num: int, title: str, body: str) -> None:
    st.markdown(
        f"""<div class="step-item">
                <div class="step-num">{num}</div>
                <div>
                    <div class="step-title">{title}</div>
                    <div class="step-body">{body}</div>
                </div>
            </div>""",
        unsafe_allow_html=True,
    )


# ============================================================================
# 4. NẠP DỮ LIỆU & MÔ HÌNH (đều cache để tránh load lại mỗi lần tương tác)
# ============================================================================


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(BASE_DIR / "data" / "processed" / "scaler.pkl")
    encoders = joblib.load(BASE_DIR / "data" / "processed" / "encoders.pkl")
    feature_names = joblib.load(BASE_DIR / "data" / "processed" / "feature_names.pkl")
    model_names = ["Logistic_Regression", "Decision_Tree", "Random_Forest", "KNN", "SVM"]
    models = {}
    for name in model_names:
        try:
            models[name] = joblib.load(BASE_DIR / "results" / "models" / f"{name}.pkl")
        except FileNotFoundError:
            pass
    return scaler, encoders, feature_names, models


@st.cache_resource
def load_dummy_model():
    try:
        return joblib.load(BASE_DIR / "results" / "models" / "Dummy_Baseline.pkl")
    except FileNotFoundError:
        return None


@st.cache_data
def load_raw_data():
    try:
        return pd.read_csv(BASE_DIR / "data" / "raw" / "du_lieu_sinh_vien_mophong.csv")
    except FileNotFoundError:
        return None


@st.cache_data
def load_comparison_table():
    try:
        return pd.read_csv(BASE_DIR / "results" / "comparison_table.csv")
    except FileNotFoundError:
        return None


@st.cache_data
def load_cv_results():
    try:
        return pd.read_csv(BASE_DIR / "results" / "cv_results.csv")
    except FileNotFoundError:
        return None


@st.cache_resource
def load_test_split():
    try:
        return joblib.load(BASE_DIR / "data" / "processed" / "train_test_data.pkl")
    except FileNotFoundError:
        return None, None, None, None


scaler, encoders, feature_names, models = load_artifacts()
dummy_model = load_dummy_model()
df_raw = load_raw_data()
comparison_df = load_comparison_table()
cv_results_df = load_cv_results()
X_train, X_test, y_train, y_test = load_test_split()

FEATURE_LABELS = {
    "GioiTinh": "Giới tính", "KhuVucSinhSong": "Khu vực sinh sống",
    "HocVanChaMe": "Học vấn cha mẹ", "HoanCanhGiaDinh": "Hoàn cảnh gia đình",
    "HocLucNamTruoc": "Học lực năm trước", "TiLeChuyenCan": "Tỉ lệ chuyên cần (%)",
    "GioTuHocMoiTuan": "Giờ tự học/tuần", "ThamGiaHocThem": "Tham gia học thêm",
    "OnThiTruoc": "Ôn thi trước", "ThoiGianMangXaHoi": "Giờ dùng MXH/ngày",
}
MODEL_LABELS = {
    "Logistic_Regression": "Logistic Regression", "Decision_Tree": "Decision Tree",
    "Random_Forest": "Random Forest", "KNN": "KNN", "SVM": "SVM",
}

best_model_name, best_model_score = None, None
# In an early-warning context, missing a student at risk is more costly than
# generating an extra warning. Select the recommended model by Recall_Truot.
selection_metric = "Recall_Truot"
if comparison_df is not None and selection_metric in comparison_df.columns:
    _cmp = comparison_df[comparison_df["Model"] != "Dummy_Baseline"]
    if len(_cmp):
        _best_row = _cmp.loc[_cmp[selection_metric].idxmax()]
        best_model_name, best_model_score = _best_row["Model"], _best_row[selection_metric]

# ============================================================================
# 5. SESSION STATE — lưu lịch sử dự đoán trong phiên làm việc hiện tại
# ============================================================================
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# ============================================================================
# 6. SIDEBAR — THƯƠNG HIỆU & ĐIỀU HƯỚNG
# ============================================================================
with st.sidebar:
    logo_col, name_col = st.columns([1, 2.4])
    with logo_col:
        logo_path = BASE_DIR / "assets" / "edupredict_ai_logo.png"
        if logo_path.exists():
            st.image(str(logo_path), width=52)
    with name_col:
        st.markdown('<p class="sidebar-brand-name">EduPredict AI</p>', unsafe_allow_html=True)
        st.markdown('<p class="sidebar-brand-sub">ACADEMIC ANALYTICS</p>', unsafe_allow_html=True)

    st.markdown(
        f'<span class="status-dot"></span><span style="color:{TEXT_MUTED};font-size:0.82rem;">'
        f'Trạng thái hệ thống: Trực tuyến</span>',
        unsafe_allow_html=True,
    )
    st.divider()

    page = st.radio(
        "Điều hướng",
        ["Giới thiệu", "Tổng quan", "Phương pháp luận",
         "Cổng tra cứu & Dự báo", "Khám phá dữ liệu", "Hiệu năng mô hình"],
        label_visibility="collapsed",
    )
    st.divider()

    st.caption("MÔ HÌNH ĐỀ XUẤT")
    if best_model_name:
        st.markdown(f"**{MODEL_LABELS.get(best_model_name, best_model_name)}**")
        st.caption(f"Recall lớp Trượt: {best_model_score:.1%}")
    else:
        st.caption("Chưa có dữ liệu đánh giá — hãy chạy notebook 04.")

    st.divider()
    st.caption("PHIÊN LÀM VIỆC")
    st.caption(f"Số lượt dự đoán đã thử: **{len(st.session_state.prediction_history)}**")

# ============================================================================
# 7. TOP BAR — hiển thị trên mọi trang
# ============================================================================
n_samples = f"{len(df_raw):,}" if df_raw is not None else "—"
model_pill = MODEL_LABELS.get(best_model_name, "—") if best_model_name else "—"
metric_pill = f" (Recall Trượt {best_model_score:.1%})" if best_model_score else ""
st.markdown(
    f'<span class="pill">Dữ liệu: {n_samples} sinh viên</span>'
    f'<span class="pill">Mô hình đề xuất: {model_pill}{metric_pill}</span>'
    f'<span class="pill">Dữ liệu mô phỏng · phiên bản 1.0</span>',
    unsafe_allow_html=True,
)
st.write("")


# ============================================================================
# TRANG 1 — GIỚI THIỆU
# ============================================================================
def render_gioi_thieu():
    st.markdown('<p class="page-title">Dự đoán khả năng Đỗ/Trượt của sinh viên</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-sub">BTL môn Trí tuệ nhân tạo — Ứng dụng Machine Learning trong giáo dục</p>',
        unsafe_allow_html=True,
    )
    st.write("")

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Số mẫu dữ liệu", f"{len(df_raw):,}" if df_raw is not None else "—",
                  "Dữ liệu mô phỏng có kiểm soát")
    with c2:
        kpi_card("Số thuộc tính đầu vào", str(len(feature_names)) if feature_names else "—",
                  "Không bao gồm điểm số gốc")
    with c3:
        kpi_card("Số mô hình so sánh", str(len(models)) + " + Baseline",
                  "5 thuật toán + mốc tham chiếu")

    st.write("")
    section_title("Bài toán được định nghĩa theo khung Task – Performance – Experience")
    tcol, pcol, ecol = st.columns(3)
    with tcol:
        st.markdown(
            f"""<div class="kpi-card">
                    <div class="kpi-label">TASK (T)</div>
                    <div style="color:{TEXT_MAIN};font-size:0.92rem;line-height:1.5;">
                    Phân loại nhị phân: dự đoán một sinh viên sẽ <b>Đỗ</b> hay <b>Trượt</b>
                    dựa trên 10 đặc trưng đầu vào.</div>
                </div>""", unsafe_allow_html=True)
    with pcol:
        st.markdown(
            f"""<div class="kpi-card">
                    <div class="kpi-label">PERFORMANCE (P)</div>
                    <div style="color:{TEXT_MAIN};font-size:0.92rem;line-height:1.5;">
                    Đo bằng Accuracy, Precision, Recall, F1-score trên tập kiểm thử độc lập —
                    ưu tiên Recall của lớp Trượt.</div>
                </div>""", unsafe_allow_html=True)
    with ecol:
        st.markdown(
            f"""<div class="kpi-card">
                    <div class="kpi-label">EXPERIENCE (E)</div>
                    <div style="color:{TEXT_MAIN};font-size:0.92rem;line-height:1.5;">
                    Học từ {len(df_raw):,} mẫu dữ liệu mô phỏng có nhãn Đỗ/Trượt, theo phương
                    pháp Supervised Learning.</div>
                </div>""" if df_raw is not None else "", unsafe_allow_html=True)

    st.write("")
    section_title("Quy trình thực hiện")
    step_item(1, "Xây dựng dữ liệu",
              "Sinh dữ liệu mô phỏng theo công thức trọng số có cơ sở khoa học, kết hợp nhiễu ngẫu nhiên.")
    step_item(2, "Tiền xử lý đúng chuẩn",
              "Chia Train/Test trước khi chuẩn hóa để tránh rò rỉ dữ liệu (data leakage).")
    step_item(3, "Huấn luyện đa mô hình",
              "5 thuật toán khác họ (Logistic Regression, Decision Tree, Random Forest, KNN, SVM) "
              "cùng 1 mốc tham chiếu Dummy Baseline.")
    step_item(4, "Đánh giá nghiêm ngặt",
              "Kết hợp đánh giá trên tập Test độc lập và 5-Fold Cross-Validation (báo cáo mean ± std).")
    step_item(5, "Triển khai ứng dụng",
              "Đóng gói mô hình thành ứng dụng web tương tác để dự đoán trên dữ liệu mới.")

    if df_raw is not None:
        st.write("")
        section_title("Xem trước dữ liệu")
        st.dataframe(df_raw.head(8), use_container_width=True)


# ============================================================================
# TRANG 2 — TỔNG QUAN
# ============================================================================
def render_tong_quan():
    st.markdown('<p class="page-title">Tổng quan dự báo học vụ</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Thống kê tổng hợp trên toàn bộ dữ liệu sinh viên</p>', unsafe_allow_html=True)
    st.write("")

    if df_raw is None:
        st.warning("Không tìm thấy dữ liệu tại `data/raw/du_lieu_sinh_vien_mophong.csv`.")
        return

    tong_sv = len(df_raw)
    ty_le_do = (df_raw["KetQua"] == "Đỗ").mean()
    so_nguy_co = (df_raw["KetQua"] == "Trượt").sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("TỔNG SỐ SINH VIÊN", f"{tong_sv:,}", "Toàn bộ dữ liệu hiện có")
    with c2:
        kpi_card("TỈ LỆ DỰ BÁO ĐỖ", f"{ty_le_do:.0%}", "Trên dữ liệu quan sát")
    with c3:
        kpi_card("SỐ SINH VIÊN NGUY CƠ", f"{so_nguy_co:,}", "Thuộc nhóm Trượt")
    with c4:
        metric_text = f"{best_model_score:.0%}" if best_model_score else "—"
        kpi_card("RECALL LỚP TRƯỢT", metric_text,
                 MODEL_LABELS.get(best_model_name, "") if best_model_name else "")

    st.write("")
    left, right = st.columns([1.3, 1])

    with left:
        section_title("Phân bố rủi ro theo khu vực sinh sống")
        risk_kv = df_raw.groupby("KhuVucSinhSong")["KetQua"].apply(lambda s: (s == "Trượt").mean())
        for label, pct in risk_kv.items():
            html_bar(label, pct, RED if pct > 0.4 else (AMBER if pct > 0.2 else GREEN))

        st.write("")
        section_title("Phân bố rủi ro theo học lực năm trước")
        order = ["Yếu", "Trung bình", "Khá", "Giỏi"]
        risk_hl = (df_raw.groupby("HocLucNamTruoc")["KetQua"]
                   .apply(lambda s: (s == "Trượt").mean()).reindex(order))
        for label, pct in risk_hl.items():
            html_bar(label, pct, RED if pct > 0.4 else (AMBER if pct > 0.2 else GREEN))

        st.write("")
        section_title("Phân bố rủi ro theo hoàn cảnh gia đình")
        risk_hc = df_raw.groupby("HoanCanhGiaDinh")["KetQua"].apply(lambda s: (s == "Trượt").mean())
        for label, pct in risk_hc.items():
            html_bar(label, pct, RED if pct > 0.4 else (AMBER if pct > 0.2 else GREEN))

    with right:
        section_title("Yếu tố ảnh hưởng nhiều nhất", "Tính từ mô hình Random Forest")
        if "Random_Forest" in models and hasattr(models["Random_Forest"], "feature_importances_"):
            imp = pd.Series(models["Random_Forest"].feature_importances_, index=feature_names)
            imp = imp.sort_values(ascending=False)
            max_imp = imp.max()
            for feat, val in imp.items():
                html_bar(FEATURE_LABELS.get(feat, feat), val / max_imp, BLUE)
        else:
            st.info("Chưa có mô hình Random Forest để phân tích.")


# ============================================================================
# TRANG 3 — PHƯƠNG PHÁP LUẬN (đối chiếu Nguyên tắc bắt buộc / Sai lầm thường gặp)
# ============================================================================
def render_phuong_phap_luan():
    st.markdown('<p class="page-title">Phương pháp luận & Tính nghiêm ngặt</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-sub">Đối chiếu quy trình thực hiện với "Nguyên tắc bắt buộc" và '
        '"Sai lầm thường gặp" đã học</p>',
        unsafe_allow_html=True,
    )
    st.write("")

    section_title("Nguyên tắc bắt buộc")
    rule_card(
        "ok", "Chỉ đo hiệu năng trên dữ liệu độc lập hoàn toàn",
        "Toàn bộ 5 mô hình được đánh giá trên tập Test (20%) chưa từng xuất hiện trong quá trình "
        "huấn luyện. Việc chuẩn hóa dữ liệu (StandardScaler) cũng chỉ fit trên tập Train, "
        "tránh rò rỉ thông tin từ tập Test.",
    )
    rule_card(
        "ok", "Chọn độ đo phù hợp với chi phí thực tế của từng loại lỗi",
        "Không dừng ở Accuracy — sử dụng Precision, Recall, F1-score, ưu tiên Recall của lớp "
        "Trượt vì bỏ sót một sinh viên có nguy cơ trượt nghiêm trọng hơn cảnh báo nhầm.",
    )
    rule_card(
        "ok", "Báo cáo cả giá trị trung bình và độ phân tán qua kiểm chứng chéo",
        "Áp dụng 5-Fold Stratified Cross-Validation trên tập Train, báo cáo Accuracy và F1-score "
        "dưới dạng mean ± std thay vì chỉ dựa vào một lần chia dữ liệu duy nhất.",
    )
    rule_card(
        "ok", "So sánh với một mốc tham chiếu đơn giản",
        "Bổ sung Dummy Baseline (luôn dự đoán lớp đa số) làm mốc so sánh — mọi mô hình đều "
        "được đối chiếu mức độ vượt trội so với mốc này trước khi kết luận là \"học được\".",
    )

    st.write("")
    section_title("Sai lầm thường gặp — đã rà soát và xử lý")
    rule_card(
        "ok", "Đánh giá trên chính tập huấn luyện",
        "Không mắc phải — toàn bộ chỉ số báo cáo đều tính trên tập Test độc lập.",
    )
    rule_card(
        "ok", "Tinh chỉnh siêu tham số trên tập kiểm thử",
        "Không mắc phải — dự án hiện chưa thực hiện Hyperparameter Tuning nên không phát sinh "
        "nguy cơ này; nếu bổ sung GridSearchCV ở giai đoạn sau sẽ chỉ thực hiện trên tập Train.",
    )
    rule_card(
        "bad", "Dùng độ chính xác trên dữ liệu mất cân bằng nghiêm trọng",
        "Từng mắc lỗi này ở phiên bản dữ liệu đầu tiên (89% Đỗ / 11% Trượt) — Accuracy cao "
        "(~89%) nhưng Recall lớp Trượt gần 0. Đã phát hiện qua classification_report và khắc "
        "phục bằng class_weight='balanced' và xây dựng lại dữ liệu với tỉ lệ 65/35 hợp lý hơn.",
    )
    rule_card(
        "ok", "Kết luận mô hình A tốt hơn B từ một lần chạy duy nhất",
        "Bổ sung Cross-Validation 5-fold để so sánh mô hình dựa trên phân phối điểm số qua "
        "nhiều lần chia dữ liệu, không chỉ dựa vào 1 con số từ 1 lần chạy.",
    )
    rule_card(
        "bad", "Tiền xử lý trên toàn bộ dữ liệu trước khi chia tập (rò rỉ dữ liệu)",
        "Phiên bản đầu tiên có lỗi này: StandardScaler được fit trên toàn bộ X trước khi "
        "train_test_split. Đã sửa: chia tập trước, sau đó chỉ fit scaler trên X_train, "
        "transform lại cho X_test.",
    )

    st.write("")
    section_title("Hạn chế còn tồn tại (tự đánh giá)")
    st.markdown(
        f"""<div class="rule-card warn">
                <div class="rule-body">
                • Dữ liệu là dữ liệu mô phỏng (synthetic), chưa được kiểm chứng trên dữ liệu
                sinh viên Việt Nam thật.<br>
                • Chưa thực hiện Hyperparameter Tuning (GridSearchCV) để tối ưu từng mô hình.<br>
                • KNN không hỗ trợ giải thích đóng góp từng đặc trưng theo từng dự đoán cụ thể.
                </div>
            </div>""",
        unsafe_allow_html=True,
    )


# ============================================================================
# TRANG 4 — CỔNG TRA CỨU & DỰ BÁO HỌC TẬP
# ============================================================================
def render_cong_tra_cuu():
    """
    Trang thiết kế theo phong cách cổng thông tin sinh viên: hồ sơ cá nhân,
    thẻ dự báo AI, phân tích nguyên nhân nguy cơ, khuyến nghị hành động, và
    mô phỏng "What-If" điều chỉnh trực tiếp các yếu tố đầu vào.

    Khác với bản demo hình ảnh gốc (theo dõi nhiều học phần riêng lẻ), do dữ
    liệu và mô hình hiện có chỉ hỗ trợ dự báo TỔNG THỂ cho 1 sinh viên (không
    có dữ liệu chi tiết theo từng môn học), toàn bộ số liệu trên trang này
    đều được tính trực tiếp từ mô hình đã huấn luyện — không có số liệu tĩnh
    giả định, đúng nguyên tắc "không đánh giá như suy luận thật trong khi
    là số liệu giả" đã đối chiếu ở trang Phương pháp luận.
    """
    st.markdown('<p class="page-title">Cổng tra cứu & Dự báo học tập</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-sub">Hồ sơ sinh viên, dự báo AI theo thời gian thực và mô phỏng cải thiện kết quả</p>',
        unsafe_allow_html=True,
    )
    st.info(
        "Kết quả là ước lượng trên dữ liệu mô phỏng, chỉ dùng để hỗ trợ trao đổi học tập. "
        "Không dùng làm căn cứ duy nhất cho quyết định học vụ."
    )
    st.write("")

    # ---------------- HỒ SƠ SINH VIÊN (tùy chọn, chỉ mang tính hiển thị) ----------------
    prof_col1, prof_col2, prof_col3 = st.columns([1.4, 1, 1])
    with prof_col1:
        ten_sv = st.text_input("Tên sinh viên (tùy chọn)", value="Sinh viên demo")
    with prof_col2:
        mssv = st.text_input("Mã số sinh viên (tùy chọn)", value="SV000000")
    with prof_col3:
        model_name = st.selectbox(
            "Mô hình dự báo", list(models.keys()),
            format_func=lambda x: MODEL_LABELS.get(x, x),
        )

    safe_name = escape(ten_sv) if ten_sv else "Sinh viên"
    safe_mssv = escape(mssv) if mssv else "—"
    st.markdown(
        f"""<div class="kpi-card" style="display:flex;align-items:center;gap:16px;margin-bottom:18px;">
                <div style="width:52px;height:52px;border-radius:50%;background:{BLUE_SOFT};
                            display:flex;align-items:center;justify-content:center;
                            font-weight:800;color:{BLUE};font-size:1.1rem;">
                    {ten_sv[:1].upper() if ten_sv else "S"}
                </div>
                <div>
                    <div style="font-weight:700;color:{TEXT_MAIN};font-size:1.05rem;">Xin chào, {safe_name}</div>
                    <div style="color:{TEXT_MUTED};font-size:0.82rem;">MSSV: {safe_mssv} · Mô hình dự báo: {MODEL_LABELS.get(model_name, model_name)}</div>
                </div>
            </div>""",
        unsafe_allow_html=True,
    )

    st.write("")
    left, right = st.columns([1.05, 1])

    # ---------------- CỘT TRÁI: HỒ SƠ HỌC TẬP (ĐẦU VÀO) ----------------
    with left:
        section_title("Hồ sơ học tập", "Điều chỉnh các giá trị bên dưới để mô phỏng tác động tới dự báo")
        col1, col2 = st.columns(2)
        with col1:
            gioi_tinh = st.selectbox("Giới tính", encoders["GioiTinh"].classes_)
            khu_vuc = st.selectbox("Khu vực sinh sống", encoders["KhuVucSinhSong"].classes_)
            hoc_van_cha_me = st.selectbox("Học vấn cha mẹ", encoders["HocVanChaMe"].classes_)
            hoan_canh = st.selectbox("Hoàn cảnh gia đình", encoders["HoanCanhGiaDinh"].classes_)
            hoc_luc = st.selectbox("Học lực năm trước", encoders["HocLucNamTruoc"].classes_)
        with col2:
            hoc_them = st.selectbox("Tham gia học thêm", encoders["ThamGiaHocThem"].classes_)
            on_thi = st.selectbox("Ôn thi trước", encoders["OnThiTruoc"].classes_)

        st.write("")
        st.markdown(f'<div class="section-title" style="margin-bottom:2px;">Mô phỏng What-If</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="section-caption">Kéo thanh trượt để xem dự báo thay đổi theo thời gian thực</div>',
                    unsafe_allow_html=True)
        chuyen_can = st.slider("Tỉ lệ chuyên cần (%)", 40.0, 100.0, 85.0, key="wi_chuyencan")
        gio_tu_hoc = st.slider("Giờ tự học/tuần", 0.0, 30.0, 9.0, key="wi_giotuhoc")
        mxh = st.slider("Giờ dùng mạng xã hội/ngày", 0.0, 10.0, 3.5, key="wi_mxh")

        luu_lich_su = st.button("Lưu kết quả này vào lịch sử", use_container_width=True)

    # ---------------- Tính toán dự báo (chạy lại mỗi khi widget thay đổi) ----------------
    raw_input = {
        "GioiTinh": gioi_tinh, "KhuVucSinhSong": khu_vuc,
        "HocVanChaMe": hoc_van_cha_me, "HoanCanhGiaDinh": hoan_canh,
        "HocLucNamTruoc": hoc_luc, "TiLeChuyenCan": chuyen_can,
        "GioTuHocMoiTuan": gio_tu_hoc, "ThamGiaHocThem": hoc_them,
        "OnThiTruoc": on_thi, "ThoiGianMangXaHoi": mxh,
    }

    def encode_input(inp: dict) -> np.ndarray:
        row = []
        for col in feature_names:
            if col in encoders:
                row.append(encoders[col].transform([str(inp[col])])[0])
            else:
                row.append(inp[col])
        return np.array(row, dtype=float).reshape(1, -1)

    model = models[model_name]
    X_new_scaled = scaler.transform(encode_input(raw_input))
    prediction = model.predict(X_new_scaled)[0]
    proba_do = model.predict_proba(X_new_scaled)[0][1] * 100 if hasattr(model, "predict_proba") else None

    if proba_do is not None:
        risk_pct = 100 - proba_do
        risk_label = "Thấp" if risk_pct < 30 else ("Trung bình" if risk_pct < 60 else "Cao")
        risk_color = GREEN if risk_pct < 30 else (AMBER if risk_pct < 60 else RED)
    else:
        risk_label, risk_color, risk_pct = "—", TEXT_MUTED, None

    # ---------------- CỘT PHẢI: THẺ DỰ BÁO AI ----------------
    with right:
        section_title("Dự báo kết quả")
        if prediction == 1:
            st.markdown(f'<div class="result-pass"><h3 style="color:{GREEN};margin:0;">DỰ BÁO: ĐỖ</h3></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-fail"><h3 style="color:{RED};margin:0;">DỰ BÁO: TRƯỢT</h3></div>',
                        unsafe_allow_html=True)

        if proba_do is not None:
            html_gauge(proba_do, GREEN if proba_do >= 50 else RED)
            st.markdown(
                f"""<div style="text-align:center;margin-top:-6px;margin-bottom:14px;">
                        <span class="pill" style="border-color:{risk_color};color:{risk_color};">
                        Chỉ số nguy cơ: {risk_label} ({risk_pct:.0f}%)</span>
                    </div>""",
                        unsafe_allow_html=True,
            )
            st.caption("Xác suất chưa được hiệu chỉnh (calibrated); hãy xem đây là tín hiệu tương đối, không phải mức độ chắc chắn.")

        # ---- Đóng góp từng yếu tố (thật, không giả định) ----
        section_title("Đóng góp từng yếu tố", "Tính bằng hệ số mô hình (Logistic/SVM) hoặc SHAP (mô hình cây)")
        contrib = None
        if model_name in ("Logistic_Regression", "SVM") and hasattr(model, "coef_"):
            contrib = pd.Series(model.coef_[0] * X_new_scaled[0], index=feature_names)
        elif model_name in ("Random_Forest", "Decision_Tree"):
            try:
                import shap
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(X_new_scaled)
                sv_class1 = sv[1][0] if isinstance(sv, list) else sv[0]
                contrib = pd.Series(sv_class1, index=feature_names)
            except ImportError:
                st.caption("Cài `pip install shap` để xem giải thích chi tiết cho mô hình cây.")

        top_negative_feat = None
        if contrib is not None:
            contrib_sorted = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(6)
            max_abs = contrib_sorted.abs().max()
            for feat, val in contrib_sorted.items():
                color = GREEN if val > 0 else RED
                html_bar(FEATURE_LABELS.get(feat, feat), abs(val) / max_abs if max_abs > 0 else 0, color)
            st.caption("Xanh: kéo tăng khả năng Đỗ · Đỏ: kéo giảm khả năng Đỗ")

            negatives = contrib[contrib < 0]
            if len(negatives):
                top_negative_feat = negatives.idxmin()
        elif model_name == "KNN":
            st.caption("Mô hình KNN không hỗ trợ giải thích chi tiết theo từng đặc trưng.")

    st.write("")
    insight_col, action_col = st.columns([1.1, 1])

    # ---------------- AI INSIGHT: mô phỏng phản thực (counterfactual) THẬT ----------------
    with insight_col:
        section_title("AI Insight — Phân tích nguyên nhân nguy cơ")
        if top_negative_feat and proba_do is not None:
            sim_input = dict(raw_input)
            note = ""
            if top_negative_feat == "ThoiGianMangXaHoi":
                sim_input["ThoiGianMangXaHoi"] = max(0.0, mxh - 2.0)
                note = "giảm 2 giờ dùng mạng xã hội/ngày"
            elif top_negative_feat == "TiLeChuyenCan":
                sim_input["TiLeChuyenCan"] = min(100.0, chuyen_can + 10.0)
                note = "tăng chuyên cần thêm 10%"
            elif top_negative_feat == "GioTuHocMoiTuan":
                sim_input["GioTuHocMoiTuan"] = gio_tu_hoc + 5.0
                note = "tăng giờ tự học thêm 5 giờ/tuần"
            else:
                note = ""

            if note:
                X_sim = scaler.transform(encode_input(sim_input))
                proba_sim = model.predict_proba(X_sim)[0][1] * 100 if hasattr(model, "predict_proba") else None
                if proba_sim is not None:
                    delta = proba_sim - proba_do
                    st.markdown(
                        f"""<div class="rule-card {'ok' if delta > 0 else 'warn'}">
                                <div class="rule-title">Yếu tố ảnh hưởng tiêu cực nhất: {FEATURE_LABELS.get(top_negative_feat, top_negative_feat)}</div>
                                <div class="rule-body">
                                Mô hình dự báo: nếu <b>{note}</b>, xác suất Đỗ ước tính thay đổi từ
                                <b>{proba_do:.0f}%</b> lên <b>{proba_sim:.0f}%</b>
                                ({'tăng' if delta > 0 else 'giảm'} {abs(delta):.0f} điểm phần trăm).
                                Đây là kết quả tính lại trực tiếp từ mô hình, không phải số liệu giả định.
                                </div>
                            </div>""",
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Không phát hiện yếu tố nào kéo giảm đáng kể khả năng Đỗ với thông tin hiện tại.")

    # ---------------- KHUYẾN NGHỊ HÀNH ĐỘNG (rule-based, minh bạch) ----------------
    with action_col:
        section_title("Khuyến nghị hành động cá nhân")
        goi_y = []
        if chuyen_can < 80:
            goi_y.append("Tỉ lệ chuyên cần dưới 80% — nên đi học đầy đủ hơn để cải thiện khả năng Đỗ.")
        if gio_tu_hoc < 8:
            goi_y.append("Giờ tự học/tuần dưới mức trung bình dữ liệu quan sát — cân nhắc tăng thời lượng tự học.")
        if mxh > 4:
            goi_y.append("Thời gian dùng mạng xã hội khá cao — đây là yếu tố có tương quan âm với kết quả học tập.")
        if "hông" in str(on_thi):  # khớp "Không ôn thi" mà không phụ thuộc thứ tự encode
            goi_y.append("Chưa ôn thi trước kỳ thi — nên tham gia ôn tập có hệ thống trước ngày thi.")
        if not goi_y:
            goi_y.append("Hồ sơ học tập hiện tại không có yếu tố rủi ro rõ rệt theo dữ liệu quan sát.")

        for g in goi_y:
            st.markdown(
                f"""<div class="rule-card warn" style="border-left-color:{BLUE};">
                        <div class="rule-body">{g}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

    # ---------------- Lưu lịch sử & xuất báo cáo ----------------
    if luu_lich_su:
        st.session_state.prediction_history.append({
            "Thời gian": datetime.now().strftime("%H:%M:%S"),
            "Mô hình": MODEL_LABELS.get(model_name, model_name),
            "Kết quả": "Đỗ" if prediction == 1 else "Trượt",
            "Xác suất Đỗ": round(proba_do / 100, 3) if proba_do is not None else None,
            "Học lực": hoc_luc, "Chuyên cần (%)": chuyen_can,
        })
        st.success("Đã lưu vào lịch sử dự đoán.")

    result_text = (
        f"CONG TRA CUU & DU BAO HOC TAP\nSinh vien: {ten_sv} ({mssv})\n"
        f"Mo hinh: {MODEL_LABELS.get(model_name, model_name)}\n"
        f"Ket qua: {'DO' if prediction == 1 else 'TRUOT'}\n"
        + (f"Xac suat Do: {proba_do:.1f}%\n" if proba_do is not None else "")
        + "\nThong tin dau vao:\n"
        + "\n".join(f"- {FEATURE_LABELS.get(k, k)}: {v}" for k, v in raw_input.items())
    )
    st.download_button("Xuất báo cáo (.txt)", result_text,
                        file_name="bao_cao_du_bao.txt", use_container_width=True)

    if st.session_state.prediction_history:
        st.write("")
        section_title("Lịch sử dự đoán trong phiên này")
        hist_df = pd.DataFrame(st.session_state.prediction_history)
        html_table(hist_df, pct_cols=["Xác suất Đỗ"])
        csv_bytes = hist_df.to_csv(index=False).encode("utf-8-sig")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("Tải lịch sử (.csv)", csv_bytes, file_name="lich_su_du_doan.csv",
                                use_container_width=True)
        with c2:
            if st.button("Xóa lịch sử", use_container_width=True):
                st.session_state.prediction_history = []
                st.rerun()



# ============================================================================
# TRANG 5 — KHÁM PHÁ DỮ LIỆU
# ============================================================================
def render_kham_pha_du_lieu():
    st.markdown('<p class="page-title">Khám phá dữ liệu</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Lọc, tìm kiếm và trực quan hóa dữ liệu sinh viên</p>', unsafe_allow_html=True)
    st.write("")

    if df_raw is None:
        st.warning("Không tìm thấy dữ liệu.")
        return

    chart_theme = dict(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color=TEXT_MAIN)

    section_title("Bộ lọc dữ liệu")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        f_ketqua = st.multiselect("Kết quả", df_raw["KetQua"].unique(), default=list(df_raw["KetQua"].unique()))
    with f2:
        f_khuvuc = st.multiselect("Khu vực", df_raw["KhuVucSinhSong"].unique(),
                                    default=list(df_raw["KhuVucSinhSong"].unique()))
    with f3:
        f_hocluc = st.multiselect("Học lực năm trước", df_raw["HocLucNamTruoc"].unique(),
                                    default=list(df_raw["HocLucNamTruoc"].unique()))
    with f4:
        f_chuyencan = st.slider("Chuyên cần tối thiểu (%)", 40, 100, 40)

    df_filtered = df_raw[
        df_raw["KetQua"].isin(f_ketqua)
        & df_raw["KhuVucSinhSong"].isin(f_khuvuc)
        & df_raw["HocLucNamTruoc"].isin(f_hocluc)
        & (df_raw["TiLeChuyenCan"] >= f_chuyencan)
    ]

    st.caption(f"Đang hiển thị {len(df_filtered):,} / {len(df_raw):,} sinh viên khớp bộ lọc")
    st.dataframe(df_filtered, use_container_width=True, height=260)
    st.download_button("Tải dữ liệu đã lọc (.csv)", df_filtered.to_csv(index=False).encode("utf-8-sig"),
                        file_name="du_lieu_da_loc.csv", use_container_width=True)

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        section_title("Tỉ lệ Đỗ / Trượt (trên dữ liệu đã lọc)")
        if len(df_filtered):
            ty_le = df_filtered["KetQua"].value_counts(normalize=True)
            pct_do = ty_le.get("Đỗ", 0) * 100
            st.markdown(
                f"""<div class="gauge-wrap">
                        <div style="width:150px;height:150px;border-radius:50%;
                                    background:conic-gradient({GREEN} 0% {pct_do}%, {RED} {pct_do}% 100%);
                                    display:flex;align-items:center;justify-content:center;">
                            <div style="width:100px;height:100px;border-radius:50%;background:{BG_PAGE};"></div>
                        </div>
                    </div>
                    <div style="text-align:center;font-size:0.85rem;color:{TEXT_MUTED};">
                        <span style="color:{GREEN};">● Đỗ {pct_do:.0f}%</span>
                        &nbsp;&nbsp;
                        <span style="color:{RED};">● Trượt {100-pct_do:.0f}%</span>
                    </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.info("Không có dữ liệu khớp bộ lọc.")
    with c2:
        fig = px.box(df_filtered, x="KetQua", y="TiLeChuyenCan", color="KetQua",
                     color_discrete_map={"Đỗ": GREEN, "Trượt": RED},
                     title="Tỉ lệ chuyên cần theo kết quả")
        fig.update_layout(**chart_theme, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.box(df_filtered, x="KetQua", y="ThoiGianMangXaHoi", color="KetQua",
                     color_discrete_map={"Đỗ": GREEN, "Trượt": RED},
                     title="Thời gian dùng mạng xã hội theo kết quả")
        fig.update_layout(**chart_theme, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        order = ["Yếu", "Trung bình", "Khá", "Giỏi"]
        fig = px.histogram(df_filtered, x="HocLucNamTruoc", color="KetQua", barmode="group",
                            category_orders={"HocLucNamTruoc": order},
                            color_discrete_map={"Đỗ": GREEN, "Trượt": RED},
                            title="Học lực năm trước và kết quả")
        fig.update_layout(**chart_theme)
        st.plotly_chart(fig, use_container_width=True)

    st.write("")
    section_title("Ma trận tương quan (biến số)")
    numeric_cols = ["TiLeChuyenCan", "GioTuHocMoiTuan", "ThoiGianMangXaHoi"]
    if len(df_filtered) > 1:
        corr = df_filtered[numeric_cols].corr()
        grid_html = ('<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px;'
                     'max-width:420px;font-size:0.78rem;text-align:center;">')
        for i in corr.index:
            for j in corr.columns:
                v = corr.loc[i, j]
                if i == j:
                    bg, fg = BLUE, "#FFFFFF"
                elif v > 0:
                    bg, fg = "rgba(76,111,255,0.18)", "#C7D2FE"
                else:
                    bg, fg = "rgba(239,68,68,0.18)", "#FCA5A5"
                grid_html += f'<div style="background:{bg};color:{fg};padding:14px 4px;border-radius:6px;">{v:.2f}</div>'
        grid_html += "</div>"
        st.markdown(grid_html, unsafe_allow_html=True)


# ============================================================================
# TRANG 6 — HIỆU NĂNG MÔ HÌNH
# ============================================================================
def render_hieu_nang_mo_hinh():
    st.markdown('<p class="page-title">Hiệu năng mô hình Machine Learning</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Đánh giá và so sánh 5 thuật toán + Dummy Baseline</p>',
                unsafe_allow_html=True)
    st.write("")

    if comparison_df is None:
        st.warning("Chưa có `results/comparison_table.csv` — hãy chạy notebook 04 trước.")
        return

    model_choices = list(models.keys())
    sel_model = st.selectbox("Chọn mô hình để xem chi tiết", model_choices,
                              format_func=lambda x: MODEL_LABELS.get(x, x))
    model = models[sel_model]

    if X_test is not None:
        from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

        y_pred = model.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            kpi_card("ACCURACY", f"{accuracy_score(y_test, y_pred):.0%}")
        with m2:
            kpi_card("PRECISION", f"{precision_score(y_test, y_pred, zero_division=0):.0%}")
        with m3:
            kpi_card("RECALL", f"{recall_score(y_test, y_pred, zero_division=0):.0%}")
        with m4:
            kpi_card("F1-SCORE", f"{f1_score(y_test, y_pred, zero_division=0):.0%}")

        st.write("")
        left, right = st.columns([1, 1])
        with left:
            section_title("Ma trận nhầm lẫn")
            r1c1, r1c2 = st.columns(2)
            with r1c1:
                cm_box("TRUE POSITIVE (Đỗ đúng)", tp, GREEN_SOFT, GREEN)
            with r1c2:
                cm_box("FALSE NEGATIVE (bỏ sót)", fn, RED_SOFT, RED)
            r2c1, r2c2 = st.columns(2)
            with r2c1:
                cm_box("FALSE POSITIVE (báo nhầm)", fp, AMBER_SOFT, AMBER)
            with r2c2:
                cm_box("TRUE NEGATIVE (Trượt đúng)", tn, BLUE_SOFT, BLUE)

        with right:
            if hasattr(model, "predict_proba"):
                from sklearn.metrics import roc_curve, auc
                y_score = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc = auc(fpr, tpr)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                          name=f"AUC = {roc_auc:.3f}", line=dict(color=BLUE, width=3)))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                          name="Ngẫu nhiên", line=dict(color=TEXT_MUTED, dash="dash")))
                fig.update_layout(title="Đường cong ROC", height=300,
                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font_color=TEXT_MAIN,
                                   xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Mô hình này không hỗ trợ tính xác suất để vẽ ROC Curve.")

    st.write("")
    section_title("Bảng so sánh trên tập Test (bao gồm Dummy Baseline)")
    display_df = comparison_df.copy()
    display_df["Model"] = display_df["Model"].map(lambda x: MODEL_LABELS.get(x, x))
    sort_col = "F1_Do" if "F1_Do" in display_df.columns else "Accuracy"
    display_df = display_df.sort_values(sort_col, ascending=False)
    pct_cols = [c for c in display_df.columns if c != "Model"]
    html_table(display_df, pct_cols=pct_cols)

    if cv_results_df is not None:
        st.write("")
        section_title("Kết quả Cross-Validation (5-fold) — Mean ± Std",
                      "Đánh giá độ ổn định của từng mô hình qua nhiều lần chia dữ liệu khác nhau")
        cv_display = cv_results_df.copy()
        cv_display["Model"] = cv_display["Model"].map(lambda x: MODEL_LABELS.get(x, x))
        cv_display["F1 (mean ± std)"] = cv_display.apply(
            lambda r: f"{r['F1_mean']:.3f} ± {r['F1_std']:.3f}", axis=1)
        cv_display["Accuracy (mean ± std)"] = cv_display.apply(
            lambda r: f"{r['Accuracy_mean']:.3f} ± {r['Accuracy_std']:.3f}", axis=1)
        html_table(cv_display[["Model", "F1 (mean ± std)", "Accuracy (mean ± std)"]])
    else:
        st.caption("Chưa có `results/cv_results.csv` — chạy notebook 03 (bản đã bổ sung Cross-Validation) để có bảng này.")


# ============================================================================
# ĐIỀU HƯỚNG CHÍNH
# ============================================================================
PAGE_RENDERERS = {
    "Giới thiệu": render_gioi_thieu,
    "Tổng quan": render_tong_quan,
    "Phương pháp luận": render_phuong_phap_luan,
    "Cổng tra cứu & Dự báo": render_cong_tra_cuu,
    "Khám phá dữ liệu": render_kham_pha_du_lieu,
    "Hiệu năng mô hình": render_hieu_nang_mo_hinh,
}

PAGE_RENDERERS[page]()
