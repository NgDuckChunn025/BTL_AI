"""One-time, lossless notebook migration. Existing notebooks are archived verbatim."""
from pathlib import Path
import shutil
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
MARKER = "edupredict_pipeline_v4"
SETUP = '''from pathlib import Path
import sys
ROOT = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent
assert (ROOT / 'src').exists(), 'Open notebook from project or notebooks directory'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from src.data_engine import PROFILE_FILE, COURSE_FILE, MODEL_DIR, FEATURES
profiles = pd.read_csv(PROFILE_FILE)
records = pd.read_csv(COURSE_FILE)
'''


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


def main():
    books = {
        "01_explore_data.ipynb": [
            md("# 01 · Bài toán và dữ liệu\nT: phân loại điểm cuối kỳ mô phỏng dưới 5. P: CV Average Precision, validation F2, test Recall/Precision/F1/AP. E: hồ sơ tự sinh, không phải dữ liệu trường. Không chứng minh hiệu quả thực tế.\n\nĐối chiếu bài giảng Chương 4: trang 7–8, 12, 19–20, 107."),
            code(SETUP),
            code("display(pd.DataFrame({'Sinh viên': [profiles.MSSV.nunique()], 'Lượt học phần': [len(records)], 'Đặc trưng': [len(FEATURES)], 'Tỷ lệ nguy cơ': [records.NguyCo.mean()]}))\ndisplay(records.head())"),
            code("display(records[FEATURES].describe(include='all').T)\ndisplay(records.isna().sum().to_frame('Missing'))\nprint('Duplicate student-course:', records.duplicated(['MSSV', 'MaHocPhan']).sum())"),
            code("records.NguyCo.value_counts().sort_index().plot.bar(title='0: không nguy cơ; 1: nguy cơ')\nplt.show()\ndisplay(records.groupby('HocPhan').NguyCo.agg(['count', 'mean']))"),
            md("## Giới hạn\nĐiểm cuối kỳ được sinh từ các đặc trưng và nhiễu; mô hình học lại một phần quy luật sinh. Không thêm thuộc tính giả để tăng vẻ thực tế. Chưa có timestamp nên không được tuyên bố đánh giá cảnh báo sớm theo thời gian. Kiểm tra ngoại lai theo miền giá trị, không tự động loại các hồ sơ yếu.")],
        "02_preprocessing.ipynb": [
            md("# 02 · Tiền xử lý và chống rò rỉ\nTách theo MSSV 60/20/20 trước mọi bước học. Pipeline fit imputer/scaler/encoder riêng trong từng fold. Bài giảng: trang 20, 31, 57, 67."),
            code(SETUP),
            code("from src.experiments import split_dataset, preprocessor\nsets = split_dataset(profiles, records)\ndisplay(pd.DataFrame({k: {'Rows': len(v), 'Students': v.MSSV.nunique(), 'Risk rate': v.NguyCo.mean()} for k,v in sets.items()}).T)\nassert not set(sets['train'].MSSV) & set(sets['test'].MSSV)\nassert not set(sets['validation'].MSSV) & set(sets['test'].MSSV)\nassert not set(sets['train'].MSSV) & set(sets['validation'].MSSV)"),
            code("assert not {'MSSV', 'HoTen', 'DiemCuoiKy', 'NguyCo'} & set(FEATURES)\nprint(FEATURES)\ndisplay(preprocessor())"),
            md("## Quy tắc\nSố: điền trung vị rồi StandardScaler. Danh mục: điền mode rồi one-hot, bỏ qua danh mục chưa gặp. Không dùng LabelEncoder để tạo thứ tự giả cho khoa/môn. Không SMOTE trước khi chia tập. Không dùng điểm cuối kỳ làm đầu vào. Notebook này không ghi đè CSV và không tạo bộ dữ liệu đã chuẩn hóa để chia CV sau đó.")],
        "03_train_models.ipynb": [
            md("# 03 · Năm mô hình và đối chứng\nLogistic Regression, Decision Tree, Random Forest, KNN, SVM + Dummy. Tìm siêu tham số trên cùng 5 fold theo nhóm. Chọn mô hình theo train CV AP; chọn ngưỡng bằng validation F2. Bài giảng: trang 46, 55–57, 61, 66.\n\nCV sau tìm kiếm có thiên lệch lạc quan; đây không phải nested CV. Test độc lập dùng báo cáo cuối."),
            code(SETUP),
            code("from src.experiments import candidates\nfor name, (_, grid) in candidates().items():\n    print(name, grid)"),
            code("# Đổi True để huấn luyện lại. Mặc định đọc kết quả đã có, tránh vô tình ghi artifact.\nRUN_TRAINING = False\nif RUN_TRAINING:\n    from src.training import train\n    metrics = train(profiles, records, folds_count=5, jobs=1)\nelse:\n    assert (MODEL_DIR / 'metrics.csv').exists(), 'Run python -m src.training first'\n    metrics = pd.read_csv(MODEL_DIR / 'metrics.csv')\ndisplay(metrics[['Model', 'CV_AP_mean', 'CV_AP_std', 'Best_Params', 'Threshold', 'Selected']])"),
            md("## Ngưỡng\nTối đa F2 trên validation trong lưới 0.05–0.95, bước 0.01; hòa chọn Precision cao hơn rồi ngưỡng cao hơn. Không refit trên validation sau khi chốt ngưỡng. Không chọn mô hình từ điểm test. Các bản chạy được giữ trong results/academic_model/runs; artifact trước được sao lưu trong archive.")],
        "04_evaluate_compare.ipynb": [
            md("# 04 · Đánh giá, phân tích lỗi và giới hạn\nChỉ đọc kết quả, không huấn luyện. Nhãn 1 = nguy cơ (khác notebook legacy: 1 = Đỗ). Bài giảng: trang 21–22, 58–67, 107–108."),
            code(SETUP),
            code("import json\nmeta = json.loads((MODEL_DIR / 'metadata.json').read_text(encoding='utf-8'))\nmetrics = pd.read_csv(MODEL_DIR / 'metrics.csv')\npredictions = pd.read_csv(MODEL_DIR / 'test_predictions.csv')\ndisplay(metrics[['Model', 'Accuracy', 'Recall_NguyCo', 'Precision_NguyCo', 'F1_NguyCo', 'F2_NguyCo', 'Average_Precision', 'ROC_AUC', 'Threshold', 'Selected']])\nprint('Selected from train CV:', meta['selected_model'])"),
            code("from src.experiments import scores\np = predictions.loc[predictions.Model.eq(meta['selected_model'])]\ndisplay(pd.Series(scores(p.NguyCo, p.Probability, meta['threshold'])))\ndisplay(p.groupby(['HocPhan', 'Error']).size().unstack(fill_value=0))\ndisplay(p.loc[p.Error.eq('FN')].head(10))"),
            code("from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay\nfig, axes = plt.subplots(1, 2, figsize=(10, 4))\nPrecisionRecallDisplay.from_predictions(p.NguyCo, p.Probability, ax=axes[0])\nRocCurveDisplay.from_predictions(p.NguyCo, p.Probability, ax=axes[1])\nplt.tight_layout()\nplt.show()"),
            code("learning = pd.read_csv(MODEL_DIR / 'learning_curve.csv')\nlearning.groupby('Fraction')[['Train_AP', 'Validation_AP']].mean().plot(marker='o', title='Learning diagnostic: ' + meta['selected_model'])\nplt.show()\ndisplay(learning.groupby('Fraction')[['Train_AP', 'Validation_AP']].agg(['mean', 'std']))"),
            md("## Biện luận bắt buộc\n1. So sánh với Dummy: Accuracy cao nhưng Recall có thể bằng 0.\n2. Phân biệt AP (Average Precision) với diện tích PR theo hình thang.\n3. Đường cong học là chẩn đoán với cấu hình đã chọn; không phải kết quả độc lập.\n4. Nếu chỉnh mô hình từ phân tích lỗi test, cần tập đánh giá mới.\n5. Không kết luận mô hình thắng tuyệt đối khi chênh lệch CV nhỏ.\n6. Dữ liệu mô phỏng, xác suất chưa kiểm định calibration, What-if không có nghĩa nhân quả.\n7. SVC probability=True còn dùng được ở sklearn 1.9 nhưng đã bị đánh dấu deprecated; cần chuyển sang calibration theo nhóm khi nâng phiên bản.\n8. Không áp dụng quyết định học vụ tự động từ bản demo.")],
    }
    for name, cells in books.items():
        destination = NOTEBOOKS / name
        if destination.exists():
            previous = nbf.read(destination, as_version=4)
            if not previous.metadata.get(MARKER):
                archived = NOTEBOOKS / "legacy" / name
                if archived.exists() and archived.read_bytes() != destination.read_bytes():
                    raise RuntimeError(f"Archive differs: {archived}; preserve manually before retrying")
                archived.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(destination, archived)
        book = nbf.v4.new_notebook(cells=cells)
        book.metadata.update({MARKER: True, "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
        nbf.validate(book)
        nbf.write(book, destination)
        print(destination)


if __name__ == "__main__":
    main()
