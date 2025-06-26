# Task 5: Personal Loan Acceptance Modeling

## 🎯 Objective

The objective of this task is to predict whether a customer will accept a personal loan offer based on their demographic and financial attributes. This is a binary classification problem using machine learning models like Logistic Regression and Decision Tree.

---

## 📁 Dataset Description

- **Dataset Name:** `bank-full.csv`
- **Source:** UCI Bank Marketing Dataset
- **Target Variable:** `y` (Loan acceptance: `yes` or `no`)
- **Features:** 
  - Demographic and financial information like `age`, `job`, `marital`, `education`, `balance`, `duration`, `campaign`, etc.

---

## ⚙️ Tools & Libraries Used

- Python
- pandas
- seaborn
- matplotlib
- scikit-learn

---

## 🧪 Approach

### 1. Load Dataset
- Loaded using `pandas` with `sep=';'` due to CSV format.

### 2. Exploratory Data Analysis (EDA)
- **Count plots** for `marital`, `job` vs. `y`
- **Histogram** for `age` vs. `y`
- Used `seaborn` for visualizing class distributions and trends.

### 3. Data Preprocessing
- Applied **Label Encoding** to convert all categorical features to numerical.
- Handled all columns detected as object types using `LabelEncoder`.

### 4. Feature Selection
- **Features (X):** All columns except `y`
- **Target (y):** Encoded column representing loan acceptance (1 for `yes`, 0 for `no`)

### 5. Train-Test Split
- Performed using an 80-20 ratio via `train_test_split`.

### 6. Model Training
- **Logistic Regression:** Used `max_iter=1000` for better convergence.
- **Decision Tree Classifier:** Trained on the same data split.

### 7. Evaluation Metrics
- Used:
  - **Classification Report**: Precision, Recall, F1-Score
  - **Confusion Matrix**

---

## 📊 Results & Insights

### 📌 Logistic Regression
- Provides a baseline classification model.
- Evaluated using precision, recall, and F1-score.

### 📌 Decision Tree
- Slightly better or more flexible on certain classes depending on depth.
- May overfit without pruning but is easy to interpret.

> **Note:** Final metrics may vary based on dataset splits and encoding results.

---

## 📂 Project Files

- `Task-05.ipynb` – Jupyter notebook containing all code and outputs
- `README.md` – This documentation

---

## ✅ Submission Checklist

- [x] Loaded and explored the dataset
- [x] Performed EDA with visualizations
- [x] Encoded categorical variables
- [x] Trained Logistic Regression and Decision Tree models
- [x] Evaluated using appropriate classification metrics
- [x] Code is clean and well-commented
- [x] Added README with clear explanation
- [x] Pushed to GitHub repository
- [x] Submitted link to Google Classroom

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/SayabArshad/Task-5-Personal-Loan-Acceptance-Prediction.git
