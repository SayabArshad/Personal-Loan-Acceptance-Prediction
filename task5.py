import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load the data
df = pd.read_csv("d:/python_ka_chilla/Internship/task 5/bank-full.csv", sep=';')
print(df.head())
print(df.info())

# Step 2: Basic Data Exploration
sns.countplot(data=df, x='marital', hue='y')
plt.title("Loan Acceptance by Marital Status")
plt.show()

sns.histplot(data=df, x='age', hue='y', multiple='stack')
plt.title("Age Distribution by Loan Acceptance")
plt.show()

sns.countplot(data=df, x='job', hue='y')
plt.xticks(rotation=45)
plt.title("Job vs Loan Acceptance")
plt.show()

# Step 3: Encode categorical variables
df_encoded = df.copy()
for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

# Step 4: Define features and target
X = df_encoded.drop('y', axis=1)
y = df_encoded['y']  # Target: 1 = accepted loan, 0 = not accepted

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Step 7: Train Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Step 8: Evaluate Models
print("Logistic Regression:")
print(classification_report(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

print("\nDecision Tree:")
print(classification_report(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
