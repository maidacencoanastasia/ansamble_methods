import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset (replace 'your_dataset.csv' with your actual dataset)
data = pd.read_csv("test_dataset.csv")

# Separate features (X) and target variable (hemoglobin)
X = data.drop("hemoglobin", axis=1)  # Features (exclude hemoglobin)
Y = data["hemoglobin"]  # Target variable

# Split the dataset into training and testing sets (e.g., 80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Choose a regression model (Linear Regression in this example)
model = LinearRegression()

# Train the regression model
model.fit(X_train, Y_train)

# Make predictions on the test data
Y_pred = model.predict(X_test)

# Evaluate the model using regression metrics
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print('Mean Squared Error (MSE): %.3f' % mse)
print('R-squared (R^2): %.3f' % r2)
