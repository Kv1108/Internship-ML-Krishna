

````markdown
# 🤖 Internship-ML-Krishna

This repository showcases a collection of applied machine learning projects developed. Each project demonstrates the use of ML models integrated with Flask to create interactive web apps for real-world problem-solving.

> 👨‍💻 Author: Krishna Viradiya    

---

## 📁 Projects Overview

### 1. 🔐 Credit Card Fraud Detection

Predicts fraudulent transactions using a Logistic Regression model trained on imbalanced financial data.

**Features:**
- Upload transaction CSV files
- Predict and highlight fraudulent transactions
- Download results with fraud labels
- User-friendly web interface

**Dataset:**  
[🔗 Credit Card Fraud Dataset (Kartik2112 - Kaggle)](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

**Tech Stack:** Python, Flask, Pandas, Scikit-learn

---

### 2. 👥 Customer Segmentation

Uses K-Means clustering to segment retail customers based on their annual income and spending score.

**Features:**
- Filter and select clustering features
- Auto-select optimal K using silhouette score
- Visualize clusters with Matplotlib & Seaborn
- Download clustered results and plots

**Dataset:**  
[🔗 Customer Segmentation Dataset (vjchoudhary7 - Kaggle)](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

**Tech Stack:** Python, Flask, Scikit-learn, Matplotlib, Seaborn

---

### 3. ✉️ Spam SMS Detection

Detects spam messages using Natural Language Processing (TF-IDF + Logistic Regression).

**Features:**
- Classify SMS as spam or ham
- View confidence score
- Minimal and responsive UI

**Dataset:**  
[🔗 SMS Spam Collection Dataset (UCI - Kaggle)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

**Tech Stack:** Python, Flask, NLTK, Scikit-learn

---

## 🚀 How to Run

1. **Clone the repository:**
```bash
git clone https://github.com/Kv1108/Internship-ML-Krishna.git
cd Internship-ML-Krishna
````

2. **Set up environment:**

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/macOS
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Run any project:**

```bash
cd <project-folder-name>
python app.py
```

Example:

```bash
cd Credit-card-fraud
python app.py
```

Access the app at: `http://localhost:5000`

---

## 📊 ML Models Used

| Project               | Model Used                   | Notes                                              |
| --------------------- | ---------------------------- | -------------------------------------------------- |
| Credit Card Fraud     | Logistic Regression          | Trained on cleaned financial transaction data      |
| Customer Segmentation | K-Means Clustering           | Clustered customers based on spending habits       |
| Spam SMS Detection    | TF-IDF + Logistic Regression | Preprocessed with tokenization & stop word removal |

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

```
