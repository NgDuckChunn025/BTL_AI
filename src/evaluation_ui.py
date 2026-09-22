"""Read-only experiment dashboard: validation exploration never changes test policy."""
import json
import pandas as pd
import streamlit as st
from src.data_engine import MODEL_DIR
from src.experiments import scores
from src.presentation import heading, download_csv


@st.cache_data(max_entries=24)
def read_report(filename, signature):
    return pd.read_csv(MODEL_DIR / filename)


def render_evaluation(metrics, metadata):
    heading("Experiment lab", "Hiệu năng & kiểm chứng mô hình", "5 mô hình · 1 đối chứng · kiểm chứng chéo theo sinh viên · test độc lập")
    st.info("Dữ liệu mô phỏng. Các kết quả đánh giá quy trình và quy luật sinh dữ liệu, chưa chứng minh hiệu quả trên sinh viên thật.")
    if metadata.get("pipeline_version") != "4.0":
        st.warning("Cần huấn luyện pipeline mới để nạp đủ báo cáo.")
        st.code(r".\venv\Scripts\python.exe -m src.training", language="powershell")
        return
    name = st.selectbox("Thuật toán", metrics.Model.tolist(), key="evaluation_model")
    row = metrics.loc[metrics.Model.eq(name)].iloc[0]
    st.badge("Đang sử dụng" if row.Selected else "Mô hình đối chiếu", color="blue")
    section = st.segmented_control("Nội dung thực nghiệm", ["Kết quả test", "So sánh CV", "Khảo sát ngưỡng", "Đường cong học", "Dữ liệu & giới hạn"],
                                   default="Kết quả test", key="evaluation_section")
    stamp = metadata["run_id"]
    if section == "So sánh CV":
        st.subheader(f"Cùng {metadata['cv_folds']} phần chia theo sinh viên")
        display = metrics[["Model", "Selected"]].copy()
        for field, label in [("AP", "Average Precision"), ("F1", "F1 nguy cơ"), ("Recall", "Recall nguy cơ")]:
            display[label + " · TB ± ĐLC"] = metrics.apply(lambda r: f"{r[f'CV_{field}_mean']:.3f} ± {r[f'CV_{field}_std']:.3f}", axis=1)
        st.dataframe(display, hide_index=True, width="stretch")
        st.caption("F1/Recall CV dùng quyết định mặc định của estimator; ngưỡng tối ưu được chọn sau trên validation. AP đo xếp hạng, không cần ngưỡng.")
        st.warning("Đây là CV dùng chọn siêu tham số, có thể lạc quan. Không phải nested CV; test độc lập mới là đánh giá cuối. Độ lệch chuẩn không phải khoảng tin cậy.")
        st.write("**Siêu tham số đã chọn**")
        st.json(json.loads(row.Best_Params))
        st.caption(f"Tìm kiếm và refit: {row.Search_Seconds:.1f} giây. So sánh thời gian phụ thuộc phần cứng.")
        download_csv("Tải chi tiết tìm kiếm", read_report("cv_results.csv", stamp), "cv_results.csv")
    elif section == "Khảo sát ngưỡng":
        st.subheader("Thử ngưỡng trên validation")
        st.caption("Chỉ phục vụ khảo sát; không thay đổi ngưỡng đã lưu hoặc kết quả test. Không chọn lại mô hình sau khi xem test.")
        threshold = st.slider("Ngưỡng thử nghiệm", .05, .95, float(row.Threshold), .01, key=f"threshold_{name}_{stamp}")
        val = read_report("validation_predictions.csv", stamp)
        val = val.loc[val.Model.eq(name)]
        measured = scores(val.NguyCo, val.Probability, threshold)
        with st.container(horizontal=True):
            for field in ["Recall_NguyCo", "Precision_NguyCo", "F2_NguyCo"]:
                st.metric(field, f"{measured[field]:.1%}", border=True)
        st.write(f"Bỏ sót: **{measured['FN']}** · Cảnh báo nhầm: **{measured['FP']}** · Ngưỡng đã lưu: **{row.Threshold:.0%}**")
        curve = read_report("threshold_validation.csv", stamp)
        curve = curve.loc[curve.Model.eq(name)]
        st.line_chart(curve.set_index("Threshold")[["Recall_NguyCo", "Precision_NguyCo", "F2_NguyCo"]])
        st.caption("F2 = 5PR / (4P + R): đặt trọng số vào Recall cao hơn F1. Đây là chính sách thí nghiệm, chưa phải tiêu chuẩn nghiệp vụ được kiểm chứng.")
    elif section == "Đường cong học":
        st.subheader(f"Chẩn đoán: {metadata['selected_model']}")
        lc = read_report("learning_curve.csv", stamp)
        summary = lc.groupby("Fraction")[["Students", "Train_AP", "Validation_AP"]].agg(["mean", "std"])
        st.line_chart(lc.groupby("Fraction")[["Train_AP", "Validation_AP"]].mean())
        st.dataframe(summary, width="stretch")
        st.caption("Trục ngang: tỷ lệ sinh viên của train fold (25/50/100%); trục dọc: Average Precision, cao hơn tốt hơn. Tổng hợp theo tỷ lệ vì số sinh viên giữa các fold hơi khác nhau. Tập nhỏ lồng nhau, lấy trọn sinh viên.")
        st.write("Train cao nhưng validation thấp: dấu hiệu khớp quá. Cả hai thấp: kiểm tra đặc trưng, độ phức tạp và nhiễu. Không kết luận chỉ từ một khoảng cách nhỏ.")
        st.caption("Dùng cấu hình đã chọn từ train CV, chỉ là chẩn đoán; không phải đánh giá độc lập hay đường cong của thuật toán đang chọn ở trên.")
        download_csv("Tải đường cong học", lc, "learning_curve.csv")
    elif section == "Dữ liệu & giới hạn":
        st.table({"Nguồn": "Tự sinh bằng src/data_engine.py; không phải dữ liệu trường học",
                  "Nhiệm vụ T": "Phân loại điểm cuối kỳ mô phỏng dưới 5/10",
                  "Độ đo P": "CV Average Precision; validation F2; test Recall/Precision/F1/F2/AP",
                  "Kinh nghiệm E": f"{metadata['dataset_rows']:,} lượt học phần; {metadata['students']:,} sinh viên mô phỏng",
                  "Lớp dương": "1 = nguy cơ; 0 = không nguy cơ",
                  "Số đặc trưng": str(len(metadata["features"])), "Mốc thời gian": "Chưa có timestamp; không chứng minh dự báo sớm theo tuần"})
        st.dataframe(pd.DataFrame(metadata["splits"]).T, width="stretch")
        st.write("Không dùng MSSV, tên sinh viên, điểm cuối kỳ hoặc nhãn làm đầu vào. Không xóa ngoại lai chỉ để nâng điểm đánh giá.")
        st.write("Không hiệu chỉnh xác suất; điểm nguy cơ không phải độ chắc chắn. What-if không chứng minh tác động nhân quả. Không dùng để tự động xử phạt sinh viên.")
        st.caption("SVC có bước ước lượng xác suất nội bộ không chia theo nhóm; toàn bộ test vẫn tách sinh viên độc lập. Cần hiệu chỉnh theo nhóm nếu mở rộng triển khai.")
        st.json(metadata)
    else:
        with st.container(horizontal=True):
            for field, label in [("Accuracy", "Accuracy"), ("Recall_NguyCo", "Recall nguy cơ"), ("Precision_NguyCo", "Precision nguy cơ"), ("F1_NguyCo", "F1 nguy cơ")]:
                st.metric(label, f"{row[field]:.1%}", border=True)
        st.caption(f"Test độc lập · Ngưỡng cố định {row.Threshold:.0%}, chọn trên validation · ROC-AUC {row.ROC_AUC:.3f} · AP {row.Average_Precision:.3f}")
        st.table(pd.DataFrame([[int(row.TP), int(row.FN)], [int(row.FP), int(row.TN)]],
                             index=["Thực tế: nguy cơ", "Thực tế: không nguy cơ"], columns=["Dự báo: nguy cơ", "Dự báo: không nguy cơ"]))
        st.write(f"Phát hiện **{int(row.TP)}/{int(row.TP+row.FN)}** ca nguy cơ; bỏ sót **{int(row.FN)}**; cảnh báo nhầm **{int(row.FP)}**.")
        pred = read_report("test_predictions.csv", stamp)
        pred = pred.loc[pred.Model.eq(name)]
        from sklearn.metrics import precision_recall_curve, roc_curve
        curve_type = st.segmented_control("Đường cong test", ["Precision–Recall", "ROC"], default="Precision–Recall")
        if curve_type == "ROC":
            x, y, _ = roc_curve(pred.NguyCo, pred.Probability)
            st.line_chart(pd.DataFrame({"FPR": x, "Recall": y}), x="FPR", y="Recall")
        else:
            p, r, _ = precision_recall_curve(pred.NguyCo, pred.Probability)
            st.line_chart(pd.DataFrame({"Recall": r, "Precision": p}), x="Recall", y="Precision")
            st.caption(f"Tỷ lệ lớp nguy cơ test: {pred.NguyCo.mean():.1%}. AP là Average Precision, không phải diện tích nội suy hình thang.")
        st.subheader("Phân tích lỗi theo học phần")
        grouped = pred.groupby(["HocPhan", "Error"]).size().unstack(fill_value=0).reindex(columns=["TP", "FP", "FN", "TN"], fill_value=0)
        st.dataframe(grouped, width="stretch")
        kind = st.selectbox("Loại hồ sơ cần xem", ["FN", "FP", "TP", "TN"], key="error_kind")
        st.dataframe(pred.loc[pred.Error.eq(kind), ["MSSV", "HocPhan", "DiemQuaTrinh", "ChuyenCan", "GioTuHoc", "Probability", "NguyCo"]], hide_index=True, width="stretch")
        st.caption("Phân tích lỗi để giải thích giới hạn; nếu dùng kết quả này thiết kế mô hình mới, phải có tập đánh giá mới trước khi tuyên bố cải thiện.")
    download_csv("Tải toàn bộ benchmark", metrics, "benchmark.csv", key="benchmark_download")
