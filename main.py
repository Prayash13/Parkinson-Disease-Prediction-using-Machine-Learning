# Parkinson's Disease Prediction using ML

# ========== Importing Required Libraries ==========
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score as ras
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

import warnings
warnings.filterwarnings('ignore')

# ========== Load Dataset ==========
df = pd.read_csv('parkinson_disease.csv')  # Make sure this file is in the same directory
print("Original Shape:", df.shape)

# ========== Data Cleaning ==========
# Group by 'id' and take mean
df = df.groupby('id').mean().reset_index()
df.drop('id', axis=1, inplace=True)

# Remove highly correlated features
columns = list(df.columns)
for col in columns:
    if col == 'class':
        continue
    filtered_columns = [col]
    for col1 in df.columns:
        if((col == col1) | (col == 'class')):
            continue
        val = df[col].corr(df[col1])
        if val > 0.7:
            if col1 in columns:
                columns.remove(col1)
        else:
            filtered_columns.append(col1)
    df = df[filtered_columns]

print("Shape after removing highly correlated features:", df.shape)

# ========== Feature Selection using Chi-Square ==========
X = df.drop('class', axis=1)
X_norm = MinMaxScaler().fit_transform(X)
selector = SelectKBest(chi2, k=30)
selector.fit(X_norm, df['class'])
filtered_columns = selector.get_support()
filtered_data = X.loc[:, filtered_columns]
filtered_data['class'] = df['class']
df = filtered_data
print("Final dataset shape after feature selection:", df.shape)

# ========== Data Imbalance Check ==========
x = df['class'].value_counts()
plt.pie(x.values, labels=x.index, autopct='%1.1f%%', startangle=90)
plt.title('Class Distribution')
plt.axis('equal')
plt.show()

# ========== Train-Test Split ==========
features = df.drop('class', axis=1)
target = df['class']

X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.2, random_state=10)
print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)

# ========== Handling Imbalanced Data ==========
ros = RandomOverSampler(sampling_strategy='minority', random_state=0)
X, Y = ros.fit_resample(X_train, Y_train)
print("Balanced train shape:", X.shape)

# ========== Model Training ==========
models = [
    LogisticRegression(),
    XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    SVC(kernel='rbf', probability=True)
]

for model in models:
    model.fit(X, Y)
    print(f"\n{model.__class__.__name__} :")

    train_preds = model.predict_proba(X)[:, 1]
    print('Training ROC AUC :', ras(Y, train_preds))

    val_preds = model.predict_proba(X_val)[:, 1]
    print('Validation ROC AUC :', ras(Y_val, val_preds))

# ========== Evaluation: Confusion Matrix and Report ==========
print("\n--- Logistic Regression Confusion Matrix and Classification Report ---")
ConfusionMatrixDisplay.from_estimator(models[0], X_val, Y_val)
plt.title("Confusion Matrix (Logistic Regression)")
plt.show()

print(classification_report(Y_val, models[0].predict(X_val)))
