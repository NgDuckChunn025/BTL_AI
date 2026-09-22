# Quy trình huấn luyện và kiểm thử để bảo vệ bài tập lớn

## Mục tiêu đúng

Mục tiêu là thực nghiệm có thể tái lập và đánh giá trung thực, không phải tìm cách
làm Accuracy tăng đến một con số định sẵn. Dữ liệu hiện tại vẫn tự sinh; kiểm thử
thành công phần mềm không chứng minh mô hình chính xác trên sinh viên thật.

## A. Chuẩn bị một lần

Mở PowerShell tại E:\BTL_AI. Dùng đúng Python của dự án, không lẫn Python hệ thống.

```powershell
Set-Location E:\BTL_AI
.\venv\Scripts\python.exe --version
.\venv\Scripts\python.exe -m pip check
```

Nếu thiếu thư viện mới chạy lệnh sau; không cần cài lại mỗi lần:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Không xóa CSV trong data/generated. Đợt nâng cấp không yêu cầu sinh lại dữ liệu.
Tự lưu một bản sao dataset/code khi bắt đầu đợt thực nghiệm để ghi nhận nguồn.
Không đưa dữ liệu cá nhân thật vào repository.

## B. Huấn luyện chính thức

```powershell
.\venv\Scripts\python.exe -m src.training --folds 5 --jobs 1
```

Đợi đủ sáu tên xuất hiện: Dummy, Logistic Regression, Decision Tree, Random Forest,
KNN, SVM; sau đó là Learning curves và bảng kết quả. Không đóng terminal giữa chừng.
Thời gian phụ thuộc máy. --jobs 1 tránh chiếm nhiều CPU/RAM.

Quy trình đang làm:

1. Tách theo sinh viên 60/20/20, không trùng MSSV giữa train/validation/test.
2. Tạo 5 fold theo nhóm trong train, dùng cùng fold cho mọi thuật toán.
3. Fit imputer/scaler/encoder trong từng Pipeline, không fit trên validation/test.
4. Tìm siêu tham số; chọn mô hình bằng Average Precision của train CV.
5. Cố định mô hình, chọn ngưỡng tối đa F2 trên validation.
6. Cố định ngưỡng rồi báo cáo test; không chọn lại bằng điểm test.
7. Lưu runs/<run_id>, sao lưu artifact trước vào archive/<run_id>.

CV sau tinh chỉnh có thể lạc quan; không phải nested CV. Độ lệch chuẩn không phải
khoảng tin cậy. Không kết luận mô hình A hơn hẳn B vì CV chỉ hơn một phần rất nhỏ.
Không chạy nhiều seed rồi chỉ giữ lần test cao nhất. Chạy lại cùng cấu hình để kiểm tra
tái lập là hợp lệ, nhưng không tạo thêm bằng chứng độc lập.

Nếu có FutureWarning về SVC probability=True: đây là cảnh báo đã biết của sklearn 1.9,
không phải lỗi huấn luyện. Đừng tự nâng sklearn trong lúc làm báo cáo; bước nâng cấp
tiếp là hiệu chỉnh xác suất theo nhóm với API mới và đánh giá lại.

## C. Kiểm thử phần mềm và artifact

Chạy từng lệnh, chỉ tiếp tục nếu lệnh trước không có AssertionError/FAILED/Traceback:

```powershell
.\venv\Scripts\python.exe scripts/check_experiments.py
.\venv\Scripts\python.exe scripts/check_new_students.py
.\venv\Scripts\python.exe scripts/check_app.py
.\venv\Scripts\python.exe -X utf8 scripts/check_notebooks.py
```

| Lệnh | Chứng minh được gì? |
|---|---|
| check_experiments | Có đủ mô hình; split không trùng; chỉ số/ngưỡng khớp prediction; Pipeline dự đoán đúng artifact |
| check_new_students | Kiểm tra CSV/biên giá trị/nhãn; không dùng kết quả thực làm đầu vào; dữ liệu cũ không bị ghi đè |
| check_app | Các trang, form, nút, mô phỏng, nhập mới, CSV và chống hiển thị kết quả cũ hoạt động |
| check_notebooks | Các cell code của bốn notebook chính chạy theo trình tự; mặc định không train lại |

PASS/OK là phần mềm chạy đúng theo các ca kiểm thử, không phải chứng nhận chất lượng
mô hình. AppTest không xác nhận màu/pixel hoặc thao tác hộp chọn file của trình duyệt.

## D. Đọc kết quả trước khi kết luận

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Mở trang Hiệu năng mô hình và kiểm tra:

- So sánh CV: đủ 5 mô hình + Dummy; trung bình và độ lệch chuẩn.
- Dữ liệu & giới hạn: đúng dataset, nhãn 1 = điểm cuối kỳ dưới 5, đúng lần train.
- Khảo sát ngưỡng: chỉ dùng validation; không thay ngưỡng theo kết quả test.
- Kết quả test: đọc đồng thời Recall, Precision, F1, AP và số FN/FP, không chỉ Accuracy.
- Đường cong học: khoảng cách train/validation có lớn không; cấu hình đã chọn nên đây là chẩn đoán.

Muốn nói hệ thống “ổn” phải có tiêu chí nghiệp vụ đặt trước: bỏ sót tối đa bao nhiêu,
cố vấn xử lý được bao nhiêu cảnh báo, chấp nhận Precision tối thiểu nào. Dự án chưa có
sự xác nhận này nên không tự đặt dấu “đạt chuẩn triển khai”. Với Precision khoảng 33%,
phần lớn cảnh báo vẫn có thể là nhầm; không được che giấu bằng Recall cao.

Nếu dùng lỗi test để chỉnh mô hình, test đó đã trở thành dữ liệu phát triển. Muốn công bố
chất lượng cải thiện phải có một tập kiểm thử mới thật sự độc lập.

## E. Nhập thử một sinh viên mới

1. Mở **Kiểm thử sinh viên mới → Một sinh viên**.
2. Chọn môn, nhập mã ẩn danh mới (không thuộc dataset cũ), khoa, nhóm và các chỉ số.
3. Nhấn **Dự đoán sinh viên mới**. Ô kết quả ghi đúng mã hồ sơ đã gửi và phiên bản model.
4. Khi chưa biết kết quả thi: chỉ đọc điểm nguy cơ, không biết dự đoán đúng/sai.
5. Khi có điểm cuối kỳ: nhập trong phần **Đối chiếu sau khi có kết quả**, nhấn đối chiếu.
6. Tải CSV/JSON nếu muốn lưu. Kết quả chỉ nằm trong phiên, không thêm vào train.

Thử chức năng có thể dùng mã NEW_2026001, GPA 2.7, quá trình 6.5, chuyên cần 85%,
LMS 12, nộp đúng hạn 85%, tự học 7 giờ, rèn luyện 75, tín chỉ tích luỹ 80.
Đây chỉ là ví dụ tự đặt. Không có nhãn “đúng” mặc định cho ví dụ này.

Một sinh viên đúng không đồng nghĩa Accuracy 100% trên quần thể.
Điểm nguy cơ 80% không có nghĩa dự đoán hồ sơ đó chính xác 80%.
Chuyên cần 85% nhập 85, không nhập 0.85. Điểm/GPA nhập đúng thang đo.

## F. Kiểm thử theo lô CSV

1. Vào **Lô CSV**, tải mẫu không nhãn hoặc mẫu có điểm đối chiếu.
2. Thay toàn bộ dòng minh hoạ bằng dữ liệu của bạn; mẫu có điểm giả lập không phải dữ liệu thật.
3. Tải file UTF-8 hoặc dán CSV, khai báo nguồn, xác nhận nhãn độc lập rồi chạy.
4. Có thể dùng dấu phẩy hoặc chấm phẩy ngăn cột; dấu chấm cho phần thập phân.

Cột bắt buộc:

```text
MSSV,MaHocPhan,GPA_TichLuy,TinChiTichLuy,DiemQuaTrinh,ChuyenCan,TruyCapLMS,TyLeNopDungHan,DiemRenLuyen,GioTuHoc,TinChi,Khoa,HocPhan,NhomHoc
```

- Không có nhãn: chỉ trả dự đoán.
- Thêm DiemCuoiKy 0–10: tự suy ra nhãn 1 khi điểm < 5.
- Hoặc thêm NguyCo 0/1. Nếu có cả hai, phải nhất quán.
- Cột nhãn đã thêm phải đủ tất cả dòng; nếu chưa có nhãn thì bỏ cột, không để trống.
- Mỗi cặp MSSV–MaHocPhan chỉ xuất hiện một lần. MSSV phải mới so với toàn bộ dataset.
- Mã môn có sẵn phải khớp tên môn/tín chỉ. Chuyên cần ngoài 0–100, GPA ngoài 0–4 bị chặn.
- Danh mục mới hoặc giá trị ngoài khoảng quan sát có cảnh báo; không cam kết mô hình tổng quát được.
- Đổi MSSV cho dòng chép nguyên đặc trưng cũ không biến nó thành mẫu độc lập.
- Chỉ một lớp: ROC-AUC/AP không được báo; chỉ số có mẫu số 0 ghi Không xác định.
- Chỉ số tính theo lượt học phần. Bốn môn của một sinh viên không phải bốn người độc lập.

Báo cáo giữ model, run_id, ngưỡng, nguồn do người dùng khai báo, kết quả từng dòng và chỉ số.
Nguồn dữ liệu không được hệ thống xác minh. Không trùng ID là điều kiện cần, không đủ
để chứng minh độc lập; vẫn phải bảo đảm nhãn không nhìn vào dự đoán và mẫu đại diện.
Lô CSV không được trộn vào metrics.csv của test gốc.

## G. Hồ sơ nên mang đi bảo vệ

- Code và requirements; hai CSV mô phỏng; seed và mô tả cách sinh dữ liệu.
- metadata.json, student_splits.csv, cv_folds.csv, cv_results.csv.
- metrics.csv, threshold_validation.csv, learning_curve.csv và phân tích FN/FP.
- Bốn notebook chính; giải thích rõ notebook legacy không dùng cho kết quả mới.
- Nếu có kiểm thử ngoài: báo cáo JSON/CSV, nguồn/thu thập/ẩn danh và cách giữ độc lập.
- Giới hạn: dữ liệu mô phỏng, không có mốc thời gian, xác suất chưa kiểm định hiệu chỉnh,
  CV không lồng nhau, chưa xác nhận hiệu quả can thiệp và chưa đủ cơ sở triển khai thực.

Tài liệu đối chiếu: bài giảng Chương 4 trang 19–22, 55–67, 107–108;
[scikit-learn: tránh rò rỉ dữ liệu](https://scikit-learn.org/stable/common_pitfalls.html)
và [ý nghĩa các độ đo](https://scikit-learn.org/stable/modules/model_evaluation.html).
