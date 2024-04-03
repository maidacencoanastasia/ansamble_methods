# Import necessary libraries
import warnings

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Data preparation
df = pd.read_csv('datasets/baseball_players.csv')

# Impute missing values with mean/median
df['Weight(pounds)'] = df['Weight(pounds)'].fillna(df['Weight(pounds)'].mean())

# Calculate BMI
df['BMI'] = df['Weight(pounds)'] / ((df['Height(inches)'] / 100) ** 2)

# Label encode positions
le = LabelEncoder()
df['Position'] = le.fit_transform(df['Position'])

# Define features and target variable
features = ['Height(inches)', 'Weight(pounds)', 'BMI']
X = df[features]
y = df['Position']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Model selection and training
models = [
    ("Logistic Regression", LogisticRegression(multi_class='multinomial')),
    ("Random Forest", RandomForestClassifier()),
    ("Gradient Boosting", GradientBoostingClassifier()),
    ("Decision Tree", DecisionTreeClassifier()),
    ("AdaBoost", AdaBoostClassifier(algorithm='SAMME')),
    ("SVM", SVC()),
    ("KNN", KNeighborsClassifier()),
    ("XGBoost", XGBClassifier())
    # ("LightGBM", LGBMClassifier())
]

for name, model in models:
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.2f}")

# models_2 = [
#     ("CatBoost", CatBoostClassifier(logging_level='Silent'))
# ]
# for name, model in models_2:
#     model.fit(X_train_scaled, y_train)
#     y_pred = model.predict(X_test_scaled)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"{name} Accuracy: {accuracy:.2f}")

# Ensemble model
ensemble = VotingClassifier(estimators=models, voting='hard')
ensemble.fit(X_train_scaled, y_train)
y_pred_ensemble = ensemble.predict(X_test_scaled)
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
print(f"Ensemble Model Accuracy: {accuracy_ensemble:.2f}")

# Evaluate ensemble model
print("Ensemble Model Report:")
print(classification_report(y_test, y_pred_ensemble))
