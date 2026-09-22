# EduPredict AI — Phòng thực nghiệm học máy và cổng học vụ

Bài tập lớn Trí tuệ nhân tạo. **Dữ liệu mô phỏng, không phải hồ sơ thực của trường.**
Phiên bản pipeline 4.0 giữ nguyên hai CSV và 12 đặc trưng hiện có; không thêm dữ liệu giả để nâng kết quả.

## Chạy ứng dụng

Từ thư mục E:\BTL_AI với môi trường đã cài:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Nếu chưa có môi trường: tạo bằng `python -m venv venv`, sau đó cài bằng
`.\venv\Scripts\python.exe -m pip install -r requirements.txt`.

Đổi sáng/tối: **⋮ → Settings → Choose app theme → Light / Dark / Use system setting**.

## Huấn luyện và kiểm tra

```powershell
.\venv\Scripts\python.exe -m src.training
.\venv\Scripts\python.exe scripts/check_experiments.py
.\venv\Scripts\python.exe scripts/check_app.py
```

Huấn luyện đọc CSV đang có, không sinh lại hoặc ghi đè dataset.
Mặc định 5 fold, một tiến trình; có thể dùng `--folds 3` để thử nhanh.
Không so sánh kết quả có số fold khác nhau như cùng một thực nghiệm.

## Bài toán T–P–E

- **T:** phân loại nguy cơ điểm thi cuối kỳ mô phỏng dưới 5/10 ở cấp sinh viên–học phần.
- **P:** chọn mô hình bằng Average Precision trên train CV; chọn ngưỡng bằng F2 trên validation; báo cáo Accuracy, Precision, Recall, F1, F2, AP, ROC-AUC trên test.
- **E:** 1.500 hồ sơ và 6.000 lượt học phần tự sinh trong data/generated.
- Nhãn **1 = nguy cơ**, **0 = không nguy cơ**. Không phải điểm tổng kết theo quy chế trường.
- Không đưa MSSV, họ tên, điểm cuối kỳ hoặc nhãn vào đặc trưng.

## Quy trình thực nghiệm

1. Kiểm tra cột, nhãn, bản ghi trùng và liên kết hồ sơ.
2. Chia theo MSSV thành train/validation/test 60/20/20: không trùng sinh viên.
3. Train CV bằng StratifiedGroupKFold: cả sáu ứng viên dùng cùng phân hoạch.
4. Imputer, StandardScaler và OneHotEncoder nằm trong Pipeline, fit lại trong mỗi fold.
5. GridSearchCV tìm siêu tham số. **Năm mô hình:** Logistic Regression, Decision Tree, Random Forest, KNN, SVM; thêm Dummy Baseline.
6. Chọn mô hình có CV Average Precision cao nhất, không chọn từ test.
7. Chọn ngưỡng riêng mỗi mô hình trên validation: tối đa F2 trong lưới 0,05–0,95, bước 0,01; hòa chọn Precision rồi ngưỡng cao hơn.
8. Giữ nguyên mô hình và ngưỡng để đánh giá test. Không refit trên validation sau khi chọn ngưỡng.
9. Lưu kết quả, đường cong học và dấu vết thí nghiệm; tự sao lưu artifact trước khi cập nhật.

CV sau tìm siêu tham số có thiên lệch lạc quan; **không phải nested CV**.
Báo cáo trung bình ± độ lệch chuẩn CV dùng cho lựa chọn; kết quả test độc lập dùng đánh giá cuối.
Không diễn giải chênh lệch nhỏ giữa các mô hình là bằng chứng vượt trội.

## Giao diện

- Hồ sơ sinh viên, tổng quan, danh sách hỗ trợ với ngưỡng lấy từ metadata.
- What-if có thao tác phân tích, lưu trong phiên, khôi phục và xuất kết quả.
- Phòng thực nghiệm: kết quả test, so sánh CV, khảo sát ngưỡng trên validation,
  đường cong học của mô hình được chọn, phân tích FN/FP và thông tin nguồn dữ liệu.
- Khảo sát ngưỡng chỉ để xem, không sửa artifact hoặc chọn ngưỡng từ test.
- Mức theo dõi bắt đầu tại 60% ngưỡng cảnh báo: quy ước UI, không phải nhóm nguy cơ đã được kiểm định.
- **Kiểm thử sinh viên mới:** form nhập hồ sơ, đối chiếu khi có điểm cuối kỳ; CSV tải lên hoặc dán trực tiếp.
  Chưa có nhãn thì không báo Accuracy. Có nhãn thì tính chỉ số riêng cho lô, không sửa benchmark.
  Chặn lỗi miền giá trị, nhãn mâu thuẫn, trùng MSSV và cảnh báo sao chép đặc trưng.

Hướng dẫn từng bước: [Huấn luyện, kiểm thử và thử sinh viên mới](docs/QUY_TRINH_HUAN_LUYEN_KIEM_THU.md).
Kiểm thử nhập liệu: `python scripts/check_new_students.py`.

## Cấu trúc

| Đường dẫn | Vai trò |
|---|---|
| app.py | Cổng học vụ và điều hướng |
| src/data_engine.py | Sinh dữ liệu demo, nạp và dự đoán |
| src/experiments.py | Tiền xử lý, chia tập, CV, tìm kiếm, ngưỡng, đánh giá |
| src/training.py | Điểm vào CLI dùng chung với notebook |
| src/evaluation_ui.py | Phòng thực nghiệm và phân tích lỗi |
| src/presentation.py, src/guide.py | Thành phần giao diện và hướng dẫn |
| notebooks/01…04 | Quy trình học tập hiện tại; notebook 03 bật RUN_TRAINING để train |
| notebooks/legacy/ | Bản gốc được giữ nguyên; không chạy để cập nhật app |
| results/academic_model/runs/ | Artifact từng lần chạy |
| results/academic_model/archive/ | Bản sao artifact trước khi cập nhật |
| docs/HUONG_DAN.md | Cách train, đọc kết quả và giới hạn |

`risk_model_v2_4.pkl` chỉ là tên tương thích; phiên bản thật nằm trong metadata.
File Gradient Boosting cũ có thể vẫn còn để bảo toàn lịch sử; không thuộc danh sách mô hình hiện tại.
Ứng dụng dùng danh sách trong metrics/metadata, không đếm file pkl.

## Giới hạn quan trọng

- Nhãn sinh bằng công thức và nhiễu: điểm đánh giá không chứng minh hiệu quả thực tế.
- Chưa có timestamp: đây chưa phải đánh giá cảnh báo sớm theo tuần hoặc kỳ học.
- Xác suất chưa được kiểm định hiệu chỉnh; What-if chỉ là độ nhạy, không phải quan hệ nhân quả.
- SVC probability=True được phiên bản sklearn 1.9 hiện tại hỗ trợ nhưng đã deprecated; bước xác suất nội bộ không chia nhóm. Test ngoài vẫn độc lập theo sinh viên. Cần calibration theo nhóm trước khi triển khai hoặc nâng sklearn.
- Không tự động xếp loại, kỷ luật hay từ chối quyền lợi sinh viên.
- Mở rộng tiếp: dữ liệu thực đã ẩn danh, đánh giá thời gian, calibration theo nhóm, kiểm định fairness.

Đối chiếu bài giảng Trần Xuân Thanh, Chương 4: trang 7–8, 19–22, 46, 55–67, 107–108.
[Tài liệu vận hành](docs/HUONG_DAN.md) · [Phân công nhóm Word](report/Phan_cong_cong_viec_EduPredict_AI.docx).
