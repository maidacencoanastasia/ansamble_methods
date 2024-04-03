import pandas as pd
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # Replace with chosen model (e.g., DecisionTreeClassifier)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score  # Optional for evaluation
from sklearn.tree import DecisionTreeClassifier

# Data preparation
df = pd.read_csv('datasets/baseball_players.csv')
# Check for missing values
print(df.isnull().sum())

# Impute missing values with mean/median
df['Weight(pounds)'] = df['Weight(pounds)'].fillna(df['Weight(pounds)'].mean())

# Calculate BMI
df['BMI'] = df['Weight(pounds)'] / ((df['Height(inches)'] / 100) ** 2)

# Label encode positions
le = LabelEncoder()
y = le.fit_transform(df['Position'])
print(le.classes_)  # This shows the unique categories mapped to numerical labels
# Assuming your features are in columns named 'Height(inches)' and 'Weight(pounds)'
features = ['Height(inches)', 'Weight(pounds)']

# Split data into features (X) and target variable (y)
X = df[features]
y = le.fit_transform(df['Position'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(df[['Height(inches)', 'Weight(pounds)']], y, test_size=0.2, random_state=42)

# Feature scaling (optional)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values)
X_test_scaled = scaler.transform(X_test.values)

# # Train the model
# model_position = LogisticRegression(multi_class='multinomial')  # Example model
# model_position.fit(X_train_scaled, y_train) # 0.27

# # Train the model
# model_position = DecisionTreeClassifier()
# model_position.fit(X_train_scaled, y_train) # 0.27

# Train Logistic Regression model (as before)
model_position = RandomForestClassifier()
model_position.fit(X_train_scaled, y_train)

# Train Decision Tree Classifier model
model_tree = DecisionTreeClassifier()
model_tree.fit(X_train_scaled, y_train)

# Create a VotingClassifier
ensemble = VotingClassifier(estimators=[('logistic', model_position), ('tree', model_tree)], voting='hard')

# Train the ensemble (voting) model
ensemble.fit(X_train_scaled, y_train) #0.27

# Use the ensemble model for prediction
predicted_positions = ensemble.predict(X_test_scaled)


# Evaluate model performance (optional)
y_pred = model_position.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")


# Function to predict position based on user input
def predict_position(height, weight):
    # Standardize user input (if scaling was used)
    user_data = scaler.transform([[height, weight]])

    # Predict using the model
    predicted_position = model_position.predict(user_data)[0]

    # Decode the predicted category
    position_names = le.inverse_transform([predicted_position])[0]

    prediction = str(position_names)
    return prediction


# Get user input for height and weight
user_height = 54 # float(input("Enter your height in inches: "))
user_weight = 204 #float(input("Enter your weight in pounds: "))

# Predict position
print(f"Predicted position for {user_weight} pounds and {user_height} inches: {predict_position(user_height, user_weight)}")
