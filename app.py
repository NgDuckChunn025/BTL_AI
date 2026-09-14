import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ============================================================
# CẤU HÌNH TRANG
# ============================================================
st.set_page_config(page_title="Dự đoán Đỗ/Trượt sinh viên", page_icon="🎓", layout="centered")

# ============================================================
# LOAD MÔ HÌNH & CÁC FILE HỖ TRỢ (đã lưu từ notebook 02, 03)
# ============================================================
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("data/processed/scaler.pkl")
    encoders = joblib.load("data/processed/encoders.pkl")
    feature_names = joblib.load("data/processed/feature_names.pkl")

    model_names = ["Logistic_Regression", "Decision_Tree", "Random_Forest", "KNN", "SVM"]
    models = {}
    for name in model_names:
        try:
            models[name] = joblib.load(f"results/models/{name}.pkl")
        except FileNotFoundError:
            pass  # bỏ qua nếu mô hình nào đó không tồn tại

    return scaler, encoders, feature_names, models


scaler, encoders, feature_names, models = load_artifacts()

# ============================================================
# GIAO DIỆN
# ============================================================
st.title("🎓 Dự đoán khả năng Đỗ/Trượt của sinh viên")
st.caption("BTL môn Trí tuệ nhân tạo — nhập thông tin sinh viên để dự đoán")

st.divider()

# ---- Chọn mô hình ----
model_name = st.selectbox("Chọn mô hình dự đoán", list(models.keys()))

st.subheader("Nhập thông tin sinh viên")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Giới tính", encoders["Gender"].classes_)
    ethnic_group = st.selectbox("Nhóm dân tộc", encoders["EthnicGroup"].classes_)
    parent_educ = st.selectbox("Trình độ học vấn cha/mẹ", encoders["ParentEduc"].classes_)
    lunch_type = st.selectbox("Loại bữa trưa (chỉ báo kinh tế gia đình)", encoders["LunchType"].classes_)
    test_prep = st.selectbox("Có ôn thi trước không", encoders["TestPrep"].classes_)

with col2:
    parent_marital = st.selectbox("Tình trạng hôn nhân cha mẹ", encoders["ParentMaritalStatus"].classes_)
    practice_sport = st.selectbox("Mức độ chơi thể thao", encoders["PracticeSport"].classes_)
    is_first_child = st.selectbox("Có phải con đầu lòng", encoders["IsFirstChild"].classes_)
    nr_siblings = st.number_input("Số anh chị em", min_value=0, max_value=10, value=1)
    transport = st.selectbox("Phương tiện đến trường", encoders["TransportMeans"].classes_)
    study_hours = st.selectbox("Số giờ tự học/tuần", encoders["WklyStudyHours"].classes_)

st.divider()

# ============================================================
# XỬ LÝ DỰ ĐOÁN
# ============================================================
if st.button("🔮 Dự đoán kết quả", use_container_width=True):

    # Đưa dữ liệu nhập vào đúng thứ tự cột như lúc train
    raw_input = {
        "Gender": gender,
        "EthnicGroup": ethnic_group,
        "ParentEduc": parent_educ,
        "LunchType": lunch_type,
        "TestPrep": test_prep,
        "ParentMaritalStatus": parent_marital,
        "PracticeSport": practice_sport,
        "IsFirstChild": is_first_child,
        "NrSiblings": nr_siblings,
        "TransportMeans": transport,
        "WklyStudyHours": study_hours,
    }

    # Mã hóa các cột dạng chữ bằng đúng encoder đã lưu lúc train
    encoded_row = []
    for col in feature_names:
        if col in encoders:
            value_encoded = encoders[col].transform([str(raw_input[col])])[0]
            encoded_row.append(value_encoded)
        else:
            encoded_row.append(raw_input[col])

    X_new = np.array(encoded_row).reshape(1, -1)
    X_new_scaled = scaler.transform(X_new)

    model = models[model_name]
    prediction = model.predict(X_new_scaled)[0]

    # Lấy xác suất nếu mô hình hỗ trợ predict_proba
    proba_text = ""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_new_scaled)[0]
        proba_pass = proba[1] * 100
        proba_text = f" (xác suất Đỗ: {proba_pass:.1f}%)"

    st.divider()
    if prediction == 1:
        st.success(f"✅ Dự đoán: **ĐỖ**{proba_text}")
    else:
        st.error(f"❌ Dự đoán: **TRƯỢT**{proba_text}")

    st.caption(f"Mô hình sử dụng: {model_name.replace('_', ' ')}")

# ============================================================
# XEM THỐNG KÊ DỮ LIỆU MẪU (tuỳ chọn, để tham khảo)
# ============================================================
with st.expander("Xem bảng so sánh hiệu suất các mô hình"):
    try:
        results_df = pd.read_csv("results/comparison_table.csv")
        st.dataframe(results_df, use_container_width=True)
    except FileNotFoundError:
        st.info("Chưa có file results/comparison_table.csv — hãy chạy notebook 04 trước.")
