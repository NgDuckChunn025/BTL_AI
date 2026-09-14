# BÁO CÁO TỔNG HỢP — BTL Trí tuệ nhân tạo
## Đề tài: Xây dựng chương trình dự đoán khả năng Đỗ/Trượt của sinh viên

> File này tổng hợp toàn bộ quá trình thực hiện đề tài tính đến thời điểm hiện tại, dùng làm
> tài liệu tham khảo khi viết báo cáo, làm slide, và chuẩn bị trả lời câu hỏi của giảng viên.

---

## MỤC LỤC
1. [Tổng quan đề tài](#1-tổng-quan-đề-tài)
2. [Nguồn dữ liệu](#2-nguồn-dữ-liệu)
3. [Giải thích các thuộc tính](#3-giải-thích-các-thuộc-tính)
4. [Quy trình thực hiện (Pipeline)](#4-quy-trình-thực-hiện-pipeline)
5. [Cấu trúc project & cách chạy](#5-cấu-trúc-project--cách-chạy)
6. [Các mô hình sử dụng](#6-các-mô-hình-sử-dụng)
7. [Kết quả thực nghiệm](#7-kết-quả-thực-nghiệm)
8. [Tiến độ đã hoàn thành](#8-tiến-độ-đã-hoàn-thành)
9. [Kế hoạch tiếp theo](#9-kế-hoạch-tiếp-theo)
10. [Hỏi & Đáp — Các câu hỏi của giảng viên](#10-hỏi--đáp--các-câu-hỏi-của-giảng-viên)
11. [Hạn chế của đề tài](#11-hạn-chế-của-đề-tài)

---

## 1. Tổng quan đề tài

**Bài toán:** Dự đoán một sinh viên sẽ **Đỗ (Pass)** hay **Trượt (Fail)** kỳ thi, dựa trên các
yếu tố cá nhân, gia đình, học tập và xã hội — **không sử dụng điểm số làm đầu vào** để mô
hình dự đoán dựa trên nguyên nhân, không phải nhìn thấy đáp án trước.

- **Dạng bài toán:** Phân loại nhị phân (Binary Classification), thuộc nhóm Supervised Learning
- **Input:** 10 đặc trưng (xem mục 3)
- **Output:** Nhãn Pass/Fail kèm xác suất dự đoán
- **Công cụ:** Python, scikit-learn, pandas, matplotlib, seaborn, VS Code, Jupyter Notebook

**Ý nghĩa thực tế:** Giúp nhà trường/giảng viên phát hiện sớm sinh viên có nguy cơ trượt để
can thiệp, hỗ trợ kịp thời (phụ đạo, tư vấn học vụ), thay vì chỉ biết kết quả sau khi thi xong.

---

## 2. Nguồn dữ liệu

| Thông tin | Chi tiết |
|---|---|
| Tên dataset | Students Exam Scores: Extended Dataset |
| Nguồn | Kaggle (kaggle.com) |
| Số lượng mẫu | 30,641 sinh viên |
| Số thuộc tính gốc | 15 cột |
| Loại dữ liệu | Dữ liệu thứ cấp, công khai |

**Vì sao chọn dataset này:**
- Số lượng mẫu lớn giúp mô hình học được quy luật tổng quát, hạn chế overfitting
- Có sẵn điểm số 3 môn để tự xây dựng nhãn (xem mục 4.3)
- Có đa dạng thuộc tính thuộc nhiều nhóm khác nhau (cá nhân, gia đình, học tập, xã hội)

**Về việc dùng dữ liệu nước ngoài thay vì dữ liệu Việt Nam:** Hiện không có dataset công khai
nào chứa dữ liệu điểm số/học vụ thật của sinh viên Việt Nam do vấn đề bảo mật thông tin nội bộ
của các trường. Đây là hạn chế khách quan, không phải do nhóm không tìm kiếm kỹ (xem thêm mục 11).

---

## 3. Giải thích các thuộc tính

| Cột | Ý nghĩa | Kiểu dữ liệu | Giá trị mẫu |
|---|---|---|---|
| `Gender` | Giới tính sinh viên | Chữ | male, female |
| `EthnicGroup` | Nhóm dân tộc (đã ẩn danh hóa thành nhóm A-E để bảo mật) | Chữ | group A...E |
| `ParentEduc` | Trình độ học vấn cao nhất của cha/mẹ | Chữ | high school, bachelor's degree... |
| `LunchType` | Loại bữa trưa ở trường — **chỉ báo gián tiếp cho điều kiện kinh tế gia đình** | Chữ | standard, free/reduced |
| `TestPrep` | Có ôn luyện trước khi thi hay không | Chữ | completed, none |
| `ParentMaritalStatus` | Tình trạng hôn nhân cha mẹ | Chữ | married, single, divorced, widowed |
| `PracticeSport` | Mức độ chơi thể thao | Chữ | never, sometimes, regularly |
| `IsFirstChild` | Có phải con đầu lòng không | Chữ | yes, no |
| `NrSiblings` | Số anh chị em ruột | Số | 0, 1, 2, 3... |
| `TransportMeans` | Phương tiện đến trường | Chữ | school_bus, private |
| `WklyStudyHours` | Số giờ tự học/tuần (đã nhóm khoảng) | Chữ | < 5hrs, 5-10hrs, > 10hrs |
| `MathScore`, `ReadingScore`, `WritingScore` | Điểm 3 môn (thang 0-100) — **chỉ dùng để tạo nhãn, không đưa vào input** | Số | 0-100 |

**Phân nhóm theo ý nghĩa:**

| Nhóm | Thuộc tính | Vai trò |
|---|---|---|
| Cá nhân | Gender, EthnicGroup, IsFirstChild, NrSiblings | Đặc điểm nhân khẩu học |
| Kinh tế - Gia đình | ParentEduc, ParentMaritalStatus, LunchType | Môi trường nuôi dạy |
| Học tập | TestPrep, WklyStudyHours | Hành vi học tập trực tiếp |
| Xã hội | PracticeSport, TransportMeans | Lối sống ngoài giờ học |

**Lưu ý về các thuộc tính "trông không liên quan":** `LunchType` và `NrSiblings` không mô tả
trực tiếp hành vi học tập, nhưng đóng vai trò biến đại diện (proxy variable) cho điều kiện
kinh tế-xã hội — cách tiếp cận phổ biến trong nghiên cứu giáo dục khi không có dữ liệu trực
tiếp về thu nhập gia đình. Mức độ ảnh hưởng thực tế của chúng được xác định qua Feature
Importance (mục 7), không dựa trên suy đoán chủ quan.

---

## 4. Quy trình thực hiện (Pipeline)

```
1. Thu thập dữ liệu → 2. Xây dựng nhãn → 3. Tiền xử lý
→ 4. Huấn luyện nhiều mô hình → 5. Đánh giá thực nghiệm → 6. Triển khai (demo)
```

### 4.1. Thu thập dữ liệu
Tải dataset CSV từ Kaggle, lưu vào `data/raw/`.

### 4.2. Khám phá dữ liệu (EDA)
Xem cấu trúc, thống kê mô tả, kiểm tra dữ liệu thiếu, vẽ biểu đồ phân bố điểm, tương quan,
so sánh theo nhóm (thực hiện ở `01_explore_data.ipynb`).

### 4.3. Xây dựng nhãn (Label Engineering) — phần quan trọng cần lưu ý
Dataset gốc **không có sẵn nhãn Đỗ/Trượt**, chỉ có điểm số 3 môn. Nhóm tự xây dựng nhãn theo
quy trình:
1. Tính điểm trung bình: `AvgScore = (MathScore + ReadingScore + WritingScore) / 3`
2. Đặt ngưỡng: `AvgScore ≥ 50` → Label = 1 (Pass); ngược lại → Label = 0 (Fail)
3. **Loại bỏ rò rỉ dữ liệu (Data Leakage):** loại 3 cột điểm gốc + AvgScore khỏi tập đặc trưng
   đầu vào (X), vì nếu giữ lại, mô hình sẽ "nhìn thấy đáp án" trực tiếp, khiến độ chính xác
   cao giả tạo, không phản ánh khả năng dự đoán thật từ các yếu tố phi học thuật.

> Ngưỡng 50/100 là giả định thiết kế của nhóm theo quy ước phổ biến, không phải chuẩn cố định,
> có thể điều chỉnh tùy tiêu chí đánh giá thực tế của từng trường.

### 4.4. Tiền xử lý dữ liệu
- Xử lý dữ liệu thiếu: điền trung vị (cột số như NrSiblings) / giá trị phổ biến nhất (cột chữ)
- Mã hóa dữ liệu dạng chữ thành số (Label Encoding)
- Chuẩn hóa dữ liệu số (StandardScaler)
- Chia tập Train (80%) / Test (20%), dùng `stratify=y` để giữ tỉ lệ Pass/Fail đồng đều giữa
  2 tập

### 4.5. Huấn luyện mô hình
Huấn luyện 4 mô hình học máy độc lập trên cùng bộ Train/Test (xem mục 6).

### 4.6. Đánh giá thực nghiệm
Đánh giá trên tập Test bằng Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC-AUC,
và phân tích Feature Importance.

### 4.7. Triển khai (đang thực hiện)
Dự kiến đóng gói mô hình tốt nhất thành ứng dụng demo (Streamlit) để nhập thông tin sinh viên
mới và xem kết quả dự đoán trực tiếp.

---

## 5. Cấu trúc project & cách chạy

```
btl-ai-du-doan-do-truot/
├── data/
│   ├── raw/                 # dữ liệu gốc (CSV tải từ Kaggle)
│   └── processed/           # dữ liệu đã tiền xử lý (.pkl)
├── notebooks/
│   ├── 01_explore_data.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_train_models.ipynb
│   └── 04_evaluate_compare.ipynb
├── results/
│   ├── models/               # mô hình đã huấn luyện (.pkl)
│   ├── figures/               # biểu đồ xuất ra
│   └── comparison_table.csv   # bảng so sánh 4 mô hình
└── README.md
```

**Thứ tự chạy bắt buộc:** `01 → 02 → 03 → 04` (mỗi file load lại kết quả `.pkl` của file trước,
không thể chạy nhảy cóc). Mỗi lần mở lại VS Code, cần bấm **Run All** lại từ đầu file vì kernel
không lưu biến giữa các phiên làm việc.

---

## 6. Các mô hình sử dụng

| Mô hình | Nguyên lý | Vai trò trong đề tài |
|---|---|---|
| Logistic Regression | Tính xác suất dựa trên tổ hợp tuyến tính của các đặc trưng | Mô hình nền (baseline), đơn giản, dễ giải thích hệ số |
| Decision Tree | Chia dữ liệu theo từng thuộc tính có mức phân tách tốt nhất | Trực quan hóa luật quyết định dạng cây |
| Random Forest | Kết hợp (ensemble) nhiều Decision Tree, biểu quyết đa số | Mô hình chính — độ chính xác cao, cho ra Feature Importance |
| KNN | Phân loại dựa trên nhãn của K điểm dữ liệu gần nhất | Đối chứng — cách tiếp cận khác (dựa khoảng cách) |

**Lưu ý về việc dùng nhiều mô hình cùng lúc:** Mỗi mô hình được huấn luyện hoàn toàn độc lập
trên cùng một bộ Train/Test, không có xung đột giữa các mô hình. Mục đích là so sánh để tìm
ra thuật toán phù hợp nhất với đặc điểm của bộ dữ liệu này.

**Lưu ý về tỉ lệ Train/Test 80/20:** Nếu dùng 100% dữ liệu để train, không còn dữ liệu độc
lập nào để kiểm tra khả năng dự đoán thật của mô hình (dễ dẫn đến overfitting — mô hình chỉ
"học thuộc" thay vì học được quy luật tổng quát). Tỉ lệ 80/20 là lựa chọn phổ biến, cân bằng
giữa việc có đủ dữ liệu để học (Train) và đủ dữ liệu để đánh giá khách quan (Test).

---

## 7. Kết quả thực nghiệm

*(Điền số liệu thật sau khi chạy xong `04_evaluate_compare.ipynb`, lấy từ
`results/comparison_table.csv`)*

| Mô hình | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | | | | |
| Decision Tree | | | | |
| Random Forest | | | | |
| KNN | | | | |

**Nhận xét:** *(điền sau khi có số liệu — nêu rõ mô hình nào tốt nhất và vì sao)*

**Feature Importance (yếu tố ảnh hưởng nhiều nhất):** *(điền top 3-5 yếu tố từ
`results/figures/feature_importance.png`)*

**Các biểu đồ minh họa (trong `results/figures/`):**
- `phan_bo_diem.png` — phân bố điểm trung bình toàn bộ sinh viên
- `correlation_heatmap.png` — tương quan giữa các biến số
- `cm_<tên mô hình>.png` — Confusion Matrix từng mô hình
- `roc_comparison.png` — so sánh ROC Curve giữa 4 mô hình
- `feature_importance.png` — mức ảnh hưởng của từng yếu tố
- `accuracy_comparison.png` — so sánh Accuracy dạng cột

---

## 8. Tiến độ đã hoàn thành

| Hạng mục | Trạng thái | Kết quả cụ thể |
|---|---|---|
| Thu thập dữ liệu | ✅ Hoàn thành | Dataset 30,641 dòng, 15 cột |
| Khám phá dữ liệu (EDA) | ✅ Hoàn thành | 4+ biểu đồ phân tích |
| Xây dựng nhãn Pass/Fail | ✅ Hoàn thành | Ngưỡng 50/100 |
| Tiền xử lý dữ liệu | ✅ Hoàn thành | Xử lý thiếu, encoding, scaling, split 80/20 |
| Huấn luyện mô hình | ✅ Hoàn thành | 4 mô hình: LogReg, Decision Tree, Random Forest, KNN |
| Đánh giá & so sánh | ✅ Hoàn thành | Accuracy, Precision, Recall, F1, Confusion Matrix, ROC |
| Xây dựng ứng dụng demo | 🔄 Đang thực hiện | |
| Viết báo cáo hoàn chỉnh | 🔄 Đang thực hiện | |

---

## 9. Kế hoạch tiếp theo

- Xây dựng ứng dụng demo (Streamlit): nhập thông tin sinh viên → dự đoán trực tiếp
- Nâng cấp mô hình: Hyperparameter Tuning (GridSearchCV), Cross-Validation, thêm thuật toán
  (SVM, Gradient Boosting)
- Giải thích mô hình sâu hơn bằng SHAP values (nếu có thời gian)
- Hoàn thiện báo cáo chi tiết và slide bảo vệ cuối kỳ

---

## 10. Hỏi & Đáp — Các câu hỏi của giảng viên

### Q1: Đề tài liên quan gì đến Machine Learning, làm về cái gì, dùng dữ liệu thế nào?
> Đề tài thuộc bài toán Phân loại (Classification) trong Supervised Learning. Cụ thể, dự đoán
> sinh viên Đỗ/Trượt dựa trên các yếu tố phi học thuật (gia đình, thói quen học tập, hoạt động
> xã hội). Dữ liệu dùng bộ "Students Exam Scores" trên Kaggle với hơn 30,000 sinh viên, có sẵn
> điểm 3 môn. Nhóm tự tạo nhãn Đỗ/Trượt từ điểm trung bình (ngưỡng ≥ 50), sau đó dùng các yếu
> tố còn lại (không gồm điểm số) để huấn luyện và so sánh nhiều thuật toán.

### Q2: Giới thiệu về bài toán và cách giải quyết bài toán?
> Xem mục 1 và mục 4 (Pipeline 6 bước) ở trên.

### Q3: Lấy dữ liệu từ đâu?
> Từ Kaggle — dataset công khai "Students Exam Scores: Extended Dataset" (xem mục 2). Do dữ
> liệu điểm số/học vụ sinh viên Việt Nam là thông tin bảo mật của từng trường, không có dataset
> công khai trong nước, nên nhóm sử dụng dữ liệu nước ngoài làm dữ liệu thực nghiệm.

### Q4: Dữ liệu tự xây dựng và quy trình xây dựng dữ liệu như thế nào?
> Dataset gốc chỉ có điểm số, không có sẵn nhãn Đỗ/Trượt. Nhóm tự xây dựng nhãn này qua quy
> trình 6 bước ở mục 4.3 và 4.4: tính điểm trung bình → đặt ngưỡng → loại bỏ rò rỉ dữ liệu →
> xử lý thiếu → mã hóa/chuẩn hóa → chia Train/Test.

### Q5: Độ chính xác khi chạy thử nghiệm là bao nhiêu, áp dụng với các thuật toán nào?
> Xem bảng kết quả ở mục 7 (điền sau khi có số liệu thật). Áp dụng 4 thuật toán: Logistic
> Regression, Decision Tree, Random Forest, KNN (xem mục 6).

### Q6: Dùng nhiều thuật toán như vậy có gây xung đột không?
> Không. Mỗi thuật toán được huấn luyện hoàn toàn độc lập trên cùng một bộ dữ liệu Train/Test,
> không ảnh hưởng lẫn nhau. Mục đích là so sánh để tìm thuật toán phù hợp nhất.

### Q7: Tại sao chỉ dùng 80% dữ liệu để Train mà không dùng 100%?
> Để giữ lại 20% làm tập kiểm tra độc lập (Test), tránh hiện tượng Overfitting — mô hình chỉ
> học thuộc dữ liệu cũ mà không thực sự dự đoán tốt trên dữ liệu mới chưa từng thấy.

### Q8: Vì sao có những thuộc tính trông không liên quan đến học tập (LunchType, NrSiblings)?
> Đây là các biến đại diện (proxy variable) cho yếu tố kinh tế-xã hội — ví dụ LunchType phản
> ánh gián tiếp điều kiện kinh tế gia đình. Mức độ ảnh hưởng thực tế được xác định qua kết quả
> Feature Importance của mô hình, không dựa trên suy đoán chủ quan.

### Q9: Sản phẩm cuối cùng của đề tài là gì?
> Một ứng dụng web (Streamlit) gồm 2 phần: (1) "Bộ não" — các mô hình học máy đã huấn luyện,
> lưu thành file, có thể tái sử dụng để dự đoán mà không cần train lại; (2) "Giao diện sử dụng"
> — cho phép người dùng xem thống kê dữ liệu mẫu, nhập thông tin sinh viên mới, và nhận kết
> quả dự đoán Đỗ/Trượt kèm mức độ tin cậy.

### Q10 (mở rộng — có thể được hỏi thêm): Làm sao biết mô hình không bị overfitting?
> So sánh Accuracy giữa tập Train và tập Test — nếu độ chênh lệch lớn (VD: Train 99%, Test
> 70%) là dấu hiệu overfitting. Ngoài ra có thể dùng K-Fold Cross-Validation để kiểm tra độ ổn
> định của mô hình qua nhiều lần chia dữ liệu khác nhau.

### Q11 (mở rộng): Nếu Accuracy giữa các mô hình gần bằng nhau thì chọn mô hình nào?
> Ưu tiên xem xét thêm Recall (đặc biệt quan trọng vì bỏ sót sinh viên nguy cơ trượt nguy hiểm
> hơn cảnh báo nhầm), độ phức tạp của mô hình (mô hình đơn giản hơn thường được ưu tiên nếu
> hiệu suất tương đương — nguyên lý Occam's Razor), và khả năng giải thích được (Random Forest
> cho Feature Importance, hỗ trợ diễn giải kết quả với người dùng cuối).

### Q12 (mở rộng): Mô hình có áp dụng được cho dữ liệu sinh viên Việt Nam không?
> Đây là hạn chế cần thừa nhận: mô hình được huấn luyện trên dữ liệu nước ngoài, các yếu tố
> văn hóa-giáo dục có thể khác biệt (VD: LunchType không phải khái niệm phổ biến ở Việt Nam).
> Muốn áp dụng thực tế cần thu thập và validate lại trên dữ liệu sinh viên Việt Nam.

---

## 11. Hạn chế của đề tài

- Dữ liệu là dữ liệu nước ngoài (Mỹ), chưa được kiểm chứng trên dữ liệu sinh viên Việt Nam
- Ngưỡng phân loại Đỗ/Trượt (50/100) là giả định của nhóm, không phải chuẩn thống nhất
- Một số thuộc tính (EthnicGroup) đã bị ẩn danh hóa, hạn chế khả năng diễn giải sâu
- Chưa thực hiện Hyperparameter Tuning đầy đủ ở giai đoạn hiện tại (dự kiến bổ sung)
- Ứng dụng demo (Streamlit) chưa hoàn thiện, đang trong quá trình xây dựng

---

*Tài liệu này được cập nhật liên tục theo tiến độ thực hiện đề tài. Vui lòng điền số liệu
thật vào mục 7 sau khi hoàn tất chạy notebook `04_evaluate_compare.ipynb`.*
