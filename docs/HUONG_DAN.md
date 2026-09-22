# Hướng dẫn EduPredict AI — Pipeline 4.0

Hướng dẫn thao tác từng bước và trang kiểm thử mới:
[Huấn luyện, kiểm thử, nhập sinh viên mới](QUY_TRINH_HUAN_LUYEN_KIEM_THU.md).

## 1. Dữ liệu giữ nguyên

Hai CSV trong data/generated vẫn là bộ 1.500 sinh viên, 6.000 lượt học phần mô phỏng.
Không thay CSV, thêm thuộc tính hoặc sửa nhãn trong lần nâng cấp này.

Đặc trưng: GPA_TichLuy, TinChiTichLuy, DiemQuaTrinh, ChuyenCan, TruyCapLMS,
TyLeNopDungHan, DiemRenLuyen, GioTuHoc, TinChi, Khoa, HocPhan, NhomHoc.
NguyCo = 1 khi điểm thi cuối kỳ mô phỏng dưới 5; không phải điểm tổng kết theo quy chế trường.
MSSV, họ tên, DiemCuoiKy và NguyCo không được làm đầu vào.

Nhiệm vụ T: phân loại nguy cơ. Độ đo P: CV AP, validation F2, test Recall/Precision/F1/F2/AP/AUC.
Kinh nghiệm E: dữ liệu tự sinh. Chưa đủ cơ sở tuyên bố dùng được cho sinh viên thực.

## 2. Chạy và huấn luyện

Chạy tại E:\BTL_AI:

```powershell
.\venv\Scripts\python.exe -m src.training
.\venv\Scripts\python.exe -m streamlit run app.py
```

Train mất thời gian hơn bản cũ vì tìm siêu tham số trên 5 fold.
`--folds 3` dành cho thử nhanh; `--jobs 1` là mặc định ổn định, tránh chiếm nhiều tài nguyên.
Lệnh train không chỉnh CSV. Kết quả cũ tự sao lưu; ứng dụng nhận phiên bản mới khi tải lại.

Nếu dùng notebook: chạy 01 → 02 → 03 → 04. Trong 03 đổi RUN_TRAINING = True để train.
Mặc định False chỉ đọc kết quả hiện tại. Không chạy notebook trong legacy để cập nhật app.

## 3. Phân hoạch và mô hình

- Train 900 sinh viên / 3.600 dòng; validation 300 / 1.200; test 300 / 1.200.
- Năm fold trong train chia theo MSSV và cố gắng giữ tỷ lệ nhãn.
- Các mô hình dùng cùng fold để so sánh công bằng.
- Điền thiếu bằng trung vị/mode; chuẩn hóa và one-hot trong từng Pipeline.
- LR: khảo sát C; cây: độ sâu và kích thước lá; rừng: độ sâu và kích thước lá;
  KNN: k và trọng số; SVM: C và kernel linear/RBF.
- Dummy luôn chọn lớp đa số, chỉ là đối chứng.
- Chọn mô hình theo Average Precision CV, không theo test.
- Chọn ngưỡng theo F2 trên validation; hòa chọn Precision rồi ngưỡng cao hơn.
- Test chỉ báo cáo sau khi cố định lựa chọn. Không huấn luyện thêm trên validation để tránh làm lệch ngưỡng.

CV sau tìm siêu tham số có thể lạc quan. Đây không phải nested CV; test là đánh giá độc lập.
Độ lệch chuẩn CV không phải khoảng tin cậy. Chênh lệch nhỏ không chứng minh vượt trội.

## 4. Đọc chỉ số

| Chỉ số | Ý nghĩa |
|---|---|
| Accuracy | (TP+TN)/N; dễ gây hiểu nhầm nếu chỉ nhìn riêng khi lệch lớp |
| Precision | TP/(TP+FP): bao nhiêu cảnh báo là đúng |
| Recall | TP/(TP+FN): tìm được bao nhiêu ca nguy cơ thật |
| F1 | 2PR/(P+R): cân bằng Precision và Recall |
| F2 | 5PR/(4P+R): đặt trọng số vào Recall cao hơn F1 |
| AP | Average Precision: tóm tắt precision theo các mức recall; không đồng nhất PR-AUC hình thang |
| ROC-AUC | Khả năng xếp hạng nguy cơ; không phải phần trăm phân loại đúng |
| TP/FN | Phát hiện đúng / bỏ sót nguy cơ |
| FP/TN | Báo nhầm / nhận diện đúng không nguy cơ |
| CV mean ± std | Trung bình ± độ lệch chuẩn qua các fold |
| Threshold | Điểm nguy cơ từ mức này trở lên được cảnh báo |

Kết quả cụ thể xem metrics.csv hoặc trang Hiệu năng; không ghi số cố định vào tài liệu để tránh lỗi thời.
Dummy có thể đạt Accuracy cao nhưng Recall bằng 0. Không chọn mô hình chỉ vì Accuracy cao.

F2 là lựa chọn thí nghiệm có chủ đích ưu tiên giảm bỏ sót, chưa được xác nhận bởi cố vấn học tập.
Hạ ngưỡng có thể tăng số cảnh báo nhầm; luôn xem cả Precision và số FP.

## 5. Các phần trên giao diện

- Kết quả test: KPI, ma trận nhầm lẫn, ROC/PR, lỗi theo môn và hồ sơ FN/FP.
- So sánh CV: bảng trung bình/độ lệch chuẩn, cấu hình được chọn, thời gian tìm kiếm.
- Khảo sát ngưỡng: tính lại trên validation; KHÔNG lưu ngưỡng mới hoặc thay đổi test.
- Đường cong học: mô hình được chọn, 25/50/100% sinh viên trong mỗi train fold;
  số cao hơn tốt hơn (AP). Train cao, validation thấp gợi ý khớp quá.
  Cấu hình đã chọn nên đồ thị là chẩn đoán, không phải đánh giá độc lập.
- Dữ liệu & giới hạn: T–P–E, phiên bản, đặc trưng, hash, tỷ lệ lớp và kích thước từng tập.

Các cảnh báo trong portal dùng threshold trong metadata.
Mức theo dõi = 0,6 × threshold, chỉ là quy ước giao diện.
Ngưỡng lọc danh sách hỗ trợ chỉ thay danh sách, không thay chính sách đã huấn luyện.

Nguy cơ = predict_proba lớp 1, chưa calibration; 1 − nguy cơ là điểm xác suất lớp 0.
Không đọc 80% là “mô hình chắc chắn đúng 80%”. Trung bình nguy cơ các môn không phải xác suất trượt cả kỳ.
What-if là độ nhạy đầu vào, không phải SHAP hay bằng chứng nhân quả.
Từ 40% xuống 30% là giảm 10 điểm phần trăm. Kịch bản chỉ lưu trong phiên; tải CSV/JSON để giữ lâu dài.

Sáng/tối: ⋮ → Settings → Choose app theme. Bảng và biểu đồ sử dụng theme gốc.

## 6. Artifact và tái lập

| File | Nội dung |
|---|---|
| metrics.csv | Test, CV summary, validation, threshold và model được chọn |
| cv_results.csv | Từng cấu hình và điểm từng fold |
| cv_folds.csv | Fold validation của mỗi sinh viên trong train |
| student_splits.csv | MSSV thuộc train/validation/test |
| threshold_validation.csv | Chỉ số trên lưới ngưỡng validation |
| validation_predictions.csv | Đầu ra validation để khảo sát |
| test_predictions.csv | Đầu ra test, đặc trưng và loại lỗi |
| learning_curve.csv | Đường cong học từng fold |
| metadata.json | Phiên bản, hash dữ liệu, thư viện, cấu hình và giới hạn |
| runs/<run_id>/ | Kết quả từng lần chạy |
| archive/<run_id>/ | Sao lưu artifact trước khi công bố lần chạy mới |

Tên risk_model_v2_4.pkl giữ để tương thích; không có nghĩa pipeline còn phiên bản 2.4.
File gradient_boosting.pkl cũ không được dùng trong benchmark mới; không xóa lịch sử.
SVC probability=True đã deprecated trong sklearn 1.9, còn chạy với requirements hiện tại;
cần calibration theo nhóm khi nâng phiên bản. Không tuyên bố xác suất đã hiệu chỉnh tốt.

## 7. Kiểm thử

```powershell
.\venv\Scripts\python.exe scripts/check_experiments.py
.\venv\Scripts\python.exe scripts/check_new_students.py
.\venv\Scripts\python.exe scripts/check_app.py
```

Kiểm tra sáu mô hình, nguồn đặc trưng, không trùng nhóm, tính lại KPI và ngưỡng,
các trang giao diện, mô phỏng và từng phần đánh giá.
AppTest không xác nhận màu sắc/pixel trên trình duyệt; vẫn nên kiểm tra Light/Dark bằng mắt.

## 8. Bảo vệ trước lớp

Bám tiêu chí trang 107 bài giảng: T–P–E và nguồn dữ liệu; Pipeline; so sánh cùng CV;
độ đo có lập luận; phân tích lỗi; tái lập và giới hạn. Báo cáo tối đa 12 trang nếu thầy áp dụng rubric này.
Ưu tiên giải thích tại sao Recall cao vẫn nhiều FP, vì sao không chọn bằng test,
và vì sao dữ liệu mô phỏng không chứng minh hiệu quả thực tế.
Không tự động xử phạt hoặc phân loại quyền lợi sinh viên bằng bản demo.
