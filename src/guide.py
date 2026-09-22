"""In-app explanations use the same meanings as the training pipeline."""
import streamlit as st

GUIDE = '''
### Dữ liệu nào đang được dùng?
App mới dùng `data/generated/student_profiles.csv` (1.500 sinh viên mô phỏng)
và `course_records.csv` (6.000 lượt sinh viên–học phần, 4 môn/sinh viên).
Dataset cũ trong `data/raw/` và notebook trong `notebooks/legacy/` chỉ để tham khảo.
Bốn notebook chính hiện dùng cùng pipeline với ứng dụng. Đợt nâng cấp giữ nguyên hai CSV và 12 đặc trưng.

### Đọc thông tin học tập
| Chỉ số | Cách hiểu |
|---|---|
| GPA tích luỹ /4 | Trung bình học tập lịch sử trên thang 4; dữ liệu hồ sơ mô phỏng. |
| Tín chỉ tích luỹ | Số tín chỉ đã tích luỹ trước kỳ đang xem. |
| Tín chỉ kỳ này | Tổng tín chỉ của các học phần đang theo dõi. |
| Điểm quá trình /10 | Đầu vào dự báo trước thi; không phải điểm cuối kỳ. |
| Chuyên cần | Phần trăm tham gia học; 90 nghĩa là 90%. |
| LMS /tuần | Số lượt truy cập hệ thống học tập trong một tuần. |
| Nộp đúng hạn | Phần trăm bài tập nộp đúng hạn. |
| Tự học | Số giờ tự học mỗi tuần. |
| Điểm rèn luyện /100 | Một đặc trưng mô phỏng, không phải điểm môn học. |

### Đọc kết quả dự báo
**Nguy cơ** là output `predict_proba` cho lớp 1: điểm thi cuối kỳ mô phỏng dưới 5/10.
Đây là định nghĩa của dataset hiện tại; chưa tích hợp công thức tổng kết học phần.
**Khả năng đạt ngưỡng** = 100% − nguy cơ. Hai số cộng lại bằng 100%.
Xác suất chưa hiệu chỉnh; ví dụ 80% không đồng nghĩa với “chắc chắn đúng 80%”.
Ngưỡng cảnh báo lấy từ metadata, được chọn trên validation bằng F2.
Mức theo dõi bắt đầu tại 60% ngưỡng đó: quy ước giao diện, không phải phân nhóm đã kiểm định.
Nguy cơ trung bình của các môn không phải xác suất trượt ít nhất một môn.

### Đọc hiệu năng mô hình
| Chỉ số | Công thức / ý nghĩa |
|---|---|
| Accuracy | (TP + TN) / tổng mẫu: tỷ lệ phân loại đúng cả hai lớp. |
| Recall nguy cơ | TP / (TP + FN): tìm được bao nhiêu trong các trường hợp thật sự nguy cơ. |
| Precision nguy cơ | TP / (TP + FP): trong các cảnh báo, bao nhiêu cảnh báo đúng. |
| F1 nguy cơ | 2 × Precision × Recall / (Precision + Recall): cân bằng hai chỉ số trên. |
| F2 nguy cơ | 5PR / (4P + R): đặt trọng số vào Recall cao hơn F1. |
| Average Precision (AP) | Tóm tắt Precision theo Recall; không đồng nhất diện tích PR nội suy hình thang. |
| CV trung bình ± độ lệch chuẩn | Mức và độ dao động qua các fold; không phải khoảng tin cậy. |
| ROC-AUC | Khả năng xếp mẫu nguy cơ cao hơn mẫu an toàn; 0,5 gần ngẫu nhiên, 1 là phân biệt hoàn hảo. Không phải phần trăm dự báo đúng. |
| TP | Thực tế nguy cơ, dự báo có nguy cơ. |
| FN | Thực tế nguy cơ nhưng mô hình bỏ sót. |
| FP | Thực tế không nguy cơ nhưng bị cảnh báo nhầm. |
| TN | Thực tế không nguy cơ và mô hình nhận diện đúng. |

Recall cao có thể đi kèm nhiều cảnh báo nhầm. Luôn đọc cả Precision, F1 và Dummy Baseline.
Các KPI benchmark tính trên test chưa dùng để chọn mô hình. Mô hình được chọn bằng
Average Precision trên train CV; ngưỡng tối đa F2 trên validation.
Sinh viên được tách riêng giữa train/validation/test (60/20/20).
CV sau tinh chỉnh có thể lạc quan; đây không phải nested CV. Không chọn mô hình từ test.
Trang Hiệu năng có so sánh CV, khảo sát ngưỡng validation, đường cong học và phân tích lỗi.

### What-if có nghĩa gì?
Nhấn **Phân tích kịch bản** để tính lại kết quả bằng các đầu vào đã chỉnh.
Phần độ nhạy thay từng yếu tố một, giữ các yếu tố khác như hồ sơ gốc.
Các chênh lệch không cộng thành tổng và không phải SHAP hay bằng chứng nhân quả.
Từ 40% xuống 30% là giảm **10 điểm phần trăm**, không phải giảm 10% tương đối.

### Sáng / tối
Mở menu **⋮ → Settings → Choose app theme**, chọn **Light**, **Dark** hoặc theo hệ thống.
Theme áp dụng đồng bộ cho chữ, bảng, widget, biểu đồ và được trình duyệt ghi nhớ.

### Huấn luyện lại
Chạy các lệnh sau tại thư mục `E:\\BTL_AI`:
```powershell
.\\venv\\Scripts\\python.exe -m src.training
.\\venv\\Scripts\\python.exe -m streamlit run app.py
```
Lệnh đầu đọc đúng hai CSV hiện tại, huấn luyện Logistic Regression, Decision Tree,
Random Forest, KNN, SVM và Dummy Baseline với 5 fold theo sinh viên,
lưu model/bảng benchmark/prediction test/metadata trong `results/academic_model/`.
Notebook chính đã đồng bộ; notebook 03 bật RUN_TRAINING để train lại.
Không chạy notebook legacy để cập nhật app. App nhận artifact mới ở lần tương tác tiếp theo.
Xem hướng dẫn chi tiết trong `docs/HUONG_DAN.md`.

### Kiểm thử sinh viên mới
Mở trang **Kiểm thử sinh viên mới** để nhập mã ẩn danh và hồ sơ chưa có trong dataset.
Chưa có kết quả cuối kỳ: chỉ có dự đoán, không thể đo Accuracy. Khi có điểm thực tế,
nhập riêng ở phần đối chiếu để xem đúng/sai; điểm này không đi vào mô hình.
Với nhiều sinh viên, tải mẫu CSV trong trang, thay dữ liệu minh hoạ rồi tải lên hoặc dán.
Có nhãn NguyCo hoặc DiemCuoiKy thì mới có chỉ số đối chiếu. Dữ liệu một lớp hoặc
mẫu số 0 sẽ có chỉ số Không xác định; không tự coi là chất lượng tốt.
Dữ liệu chỉ xử lý trong phiên và xuất theo yêu cầu, không ghi đè dataset hoặc benchmark.
Không dùng vài hồ sơ tự đặt để kết luận hệ thống đạt chuẩn thực tế.

### Kiểm tra sau huấn luyện
```powershell
.\\venv\\Scripts\\python.exe scripts/check_experiments.py
.\\venv\\Scripts\\python.exe scripts/check_new_students.py
.\\venv\\Scripts\\python.exe scripts/check_app.py
.\\venv\\Scripts\\python.exe -X utf8 scripts/check_notebooks.py
```
PASS/OK là kiểm thử phần mềm thành công, không phải chứng nhận độ chính xác mô hình.
Xem `docs/QUY_TRINH_HUAN_LUYEN_KIEM_THU.md` cho quy trình và checklist bảo vệ.
'''


def render_guide():
    st.markdown(GUIDE)
