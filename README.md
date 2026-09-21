# EduPredict AI

Ứng dụng Streamlit minh hoạ bài toán dự báo **Đỗ / Trượt** của sinh viên bằng các yếu tố về thói quen học tập và bối cảnh cá nhân. Đây là sản phẩm bài tập lớn môn Trí tuệ nhân tạo, được thiết kế để trình bày trọn vẹn quy trình: dữ liệu → tiền xử lý → huấn luyện → đánh giá → trải nghiệm dự báo.

> Lưu ý: phiên bản hiện tại dùng **1.500 bản ghi dữ liệu mô phỏng**. Kết quả dự báo chỉ có giá trị minh hoạ/hỗ trợ trao đổi học tập, không được dùng như căn cứ duy nhất để đưa ra quyết định học vụ.

## Chức năng

- Tổng quan các nhóm rủi ro theo khu vực, học lực năm trước và hoàn cảnh gia đình.
- Dự báo cá nhân với mô phỏng What-if cho chuyên cần, tự học và thời gian mạng xã hội.
- Đề xuất hành động dựa trên hồ sơ đầu vào và tính toán phản thực từ mô hình.
- Khám phá dữ liệu bằng bộ lọc, bảng dữ liệu, biểu đồ và xuất CSV.
- So sánh 5 mô hình với Dummy Baseline, tập kiểm thử độc lập và 5-fold cross-validation.
- Xuất báo cáo dự báo và lịch sử dự báo trong phiên làm việc.

## Dữ liệu

File dữ liệu: `data/raw/du_lieu_sinh_vien_mophong.csv`

| Nội dung | Giá trị |
| --- | --- |
| Số bản ghi | 1.500 sinh viên |
| Thuộc tính đầu vào | 10 |
| Nhãn | `Đỗ`, `Trượt` |
| Tỷ lệ nhãn hiện tại | 65% Đỗ, 35% Trượt |
| Kiểu dữ liệu | Mô phỏng có kiểm soát |

Các biến đầu vào gồm giới tính, khu vực sống, học vấn cha mẹ, hoàn cảnh gia đình, học lực năm trước, chuyên cần, giờ tự học, học thêm, ôn thi và thời gian dùng mạng xã hội. Không dùng điểm số gốc làm feature dự báo.

Khi thay bằng dữ liệu thực tế, cần thực hiện ẩn danh, có cơ sở pháp lý/đồng thuận phù hợp, đánh giá bias theo nhóm và huấn luyện lại toàn bộ pipeline. Không được dùng trực tiếp các artifact hiện tại cho dữ liệu mới.

## Mô hình và cách đánh giá

| Mô hình | Vai trò |
| --- | --- |
| Dummy Baseline | Mốc tối thiểu, luôn dự đoán lớp phổ biến nhất |
| Logistic Regression | Baseline có thể giải thích |
| Decision Tree | Mô hình luật trực quan |
| Random Forest | Ensemble và feature importance |
| KNN | So sánh theo khoảng cách |
| Linear SVM | Mô hình biên phân tách tuyến tính |

Quy trình đánh giá:

1. Chia stratified train/test theo tỷ lệ 80/20.
2. Fit `StandardScaler` chỉ trên train để tránh data leakage.
3. Đánh giá Accuracy, Precision, Recall, F1 cho cả hai lớp trên test độc lập.
4. Thực hiện 5-fold Stratified Cross-Validation trên tập train và báo cáo trung bình ± độ lệch chuẩn.

Trong bối cảnh cảnh báo sớm, chỉ số quan trọng là **Recall lớp Trượt**: bỏ sót một sinh viên có nguy cơ thường gây hại hơn một cảnh báo thừa. Với artifact hiện tại, SVM có Recall lớp Trượt cao nhất (83,8%); Random Forest và SVM đồng hạng Accuracy cao nhất (84,7%). Các chỉ số này chỉ phản ánh dữ liệu mô phỏng hiện tại.

## Cấu trúc dự án

```text
BTL_AI/
├── app.py                         # Ứng dụng Streamlit
├── data/
│   ├── raw/du_lieu_sinh_vien_mophong.csv
│   └── processed/                 # scaler, encoder, feature names, train/test split
├── notebooks/
│   ├── 01_explore_data (1).ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_train_models.ipynb
│   └── 04_evaluate_compare.ipynb
├── results/
│   ├── models/                    # các model đã huấn luyện
│   ├── figures/                   # biểu đồ EDA và đánh giá
│   ├── comparison_table.csv
│   ├── cv_results.csv
│   └── final_summary_table.csv
├── assets/edupredict_ai_logo.png
└── requirements.txt
```

## Cài đặt và chạy

Yêu cầu Python 3.11–3.13.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Sau khi chạy, mở địa chỉ Streamlit hiển thị trong terminal (thường là `http://localhost:8501`).

## Tái tạo pipeline

Chạy notebook theo đúng thứ tự dưới đây từ thư mục `notebooks/` hoặc điều chỉnh đường dẫn tương ứng:

1. `01_explore_data (1).ipynb`
2. `02_preprocessing.ipynb`
3. `03_train_models.ipynb`
4. `04_evaluate_compare.ipynb`

Các notebook sau phụ thuộc artifact được tạo bởi notebook trước. Khi thay đổi dữ liệu, hãy chạy lại toàn bộ chuỗi để bảo đảm encoder, scaler, model và các bảng kết quả cùng một phiên bản dữ liệu.

## Giới hạn và hướng phát triển

- Dữ liệu mô phỏng chưa chứng minh khả năng tổng quát trên sinh viên thật hoặc bối cảnh Việt Nam.
- Xác suất dự báo chưa được calibration; nên hiểu như tín hiệu tương đối.
- Chưa có tuning lồng (nested CV), phân tích fairness hoặc theo dõi drift dữ liệu.
- Giải thích SHAP cho mô hình cây là tuỳ chọn và chưa được đóng gói mặc định.

Các ưu tiên tiếp theo là: đưa preprocessing và model vào `sklearn.Pipeline`/`ColumnTransformer`; tối ưu threshold theo chi phí bỏ sót lớp Trượt; calibration xác suất; model/data card; kiểm tra fairness; và tách `app.py` thành các module `src/` khi sản phẩm mở rộng.
