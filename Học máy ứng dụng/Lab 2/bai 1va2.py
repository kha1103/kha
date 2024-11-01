import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import os

# --- Bài 1: Bernoulli và Multinomial Naive Bayes ---
def naive_bayes_classification():
    # Tải tập dữ liệu
    dataset = "D:/Huỳnh Chí Kha - 2174802010905 - lab2/Education.csv"
    data = pd.read_csv(dataset)

    # LabelBinarizer chuyển đổi nhãn 'Dương'/'Âm'
    label_binarizer = LabelBinarizer()
    y = label_binarizer.fit_transform(data['Label']).ravel()

    # Sử dụng TfidfVectorizer để trích xuất tính năng
    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform(data['Text'])

    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Bernoulli Naive Bayes
    bernoulli_nb = BernoulliNB()
    bernoulli_nb.fit(X_train, y_train)
    y_pred_bernoulli = bernoulli_nb.predict(X_test)
    accuracy_bernoulli = accuracy_score(y_test, y_pred_bernoulli) * 100
    report_bernoulli = classification_report(y_test, y_pred_bernoulli)

    # Multinomial Naive Bayes
    multinomial_nb = MultinomialNB()
    multinomial_nb.fit(X_train, y_train)
    y_pred_multinomial = multinomial_nb.predict(X_test)
    accuracy_multinomial = accuracy_score(y_test, y_pred_multinomial) * 100
    report_multinomial = classification_report(y_test, y_pred_multinomial)

    return accuracy_bernoulli, report_bernoulli, accuracy_multinomial, report_multinomial

# --- Bài 2: Gaussian Naive Bayes ---
def gaussian_naive_bayes_classification():
    # Tải tập dữ liệu
    dataset = "D:/Huỳnh Chí Kha - 2174802010905 - lab2/drug200.csv"
    data = pd.read_csv(dataset)

    # Mã hóa 'Sex', 'BP', 'Cholesterol' và 'Drug'
    label_encoder = LabelEncoder()
    data['Sex'] = label_encoder.fit_transform(data['Sex'])
    data['BP'] = label_encoder.fit_transform(data['BP'])
    data['Cholesterol'] = label_encoder.fit_transform(data['Cholesterol'])
    data['Drug'] = label_encoder.fit_transform(data['Drug'])

    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
    y = data['Drug']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Áp dụng Gaussian Naive Bayes
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)

    # Đánh giá mô hình
    accuracy_gaussian = accuracy_score(y_test, y_pred) * 100
    report_gaussian = classification_report(y_test, y_pred)

    return accuracy_gaussian, report_gaussian

# --- Ghi kết quả của cả hai bài vào file HTML ---
def write_results_to_html(template_file, output_file, naive_bayes_results, gaussian_results):
    accuracy_bernoulli, report_bernoulli, accuracy_multinomial, report_multinomial = naive_bayes_results
    accuracy_gaussian, report_gaussian = gaussian_results

    # Đọc nội dung file template.html
    with open(template_file, 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Thay thế các placeholder bằng kết quả thực tế
    html_content = html_content.replace("{{accuracy_bernoulli}}", f"{accuracy_bernoulli:.2f}")
    html_content = html_content.replace("{{report_bernoulli}}", report_bernoulli)
    html_content = html_content.replace("{{accuracy_multinomial}}", f"{accuracy_multinomial:.2f}")
    html_content = html_content.replace("{{report_multinomial}}", report_multinomial)
    html_content = html_content.replace("{{accuracy_gaussian}}", f"{accuracy_gaussian:.2f}")
    html_content = html_content.replace("{{report_gaussian}}", report_gaussian)

    # Ghi kết quả vào file HTML
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Kết quả đã được ghi vào file {output_file}.")

# --- Chạy cả hai bài toán và ghi kết quả vào file HTML ---
if __name__ == "__main__":
    # Bài 1: Kết quả từ Bernoulli và Multinomial Naive Bayes
    naive_bayes_results = naive_bayes_classification()

    # Bài 2: Kết quả từ Gaussian Naive Bayes
    gaussian_results = gaussian_naive_bayes_classification()

    # Ghi kết quả vào file HTML
    write_results_to_html('template.html', 'index.html', naive_bayes_results, gaussian_results)

    # Tự động mở file index.html trên Windows
    os.system("start index.html")
    # Trên macOS, sử dụng `os.system("open index.html")`
