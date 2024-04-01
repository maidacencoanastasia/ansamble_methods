import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset (replace 'your_dataset.csv' with your actual dataset)
data = pd.read_csv("test_dataset.csv")

# Separate features (X) and target variable (hemoglobin)
X = data.drop("hemoglobin", axis=1)  # Features (exclude hemoglobin)
Y = data["hemoglobin"]  # Target variable

# Split the dataset into training and testing sets (e.g., 80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)


# Create a Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=1)

# Train the Random Forest model
rf_model.fit(X_train, Y_train)

# Make predictions on the test data
rf_Y_pred = rf_model.predict(X_test)

# Evaluate the Random Forest model using regression metrics
rf_mse = mean_squared_error(Y_test, rf_Y_pred)
rf_r2 = r2_score(Y_test, rf_Y_pred)

print('Random Forest Mean Squared Error (MSE): %.3f' % rf_mse)
print('Random Forest R-squared (R^2): %.3f' % rf_r2)
# Create a Gradient Boosting Regressor model
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=1)

# Train the Gradient Boosting model
gb_model.fit(X_train, Y_train)

# Make predictions on the test data
gb_Y_pred = gb_model.predict(X_test)

# Evaluate the Gradient Boosting model using regression metrics
gb_mse = mean_squared_error(Y_test, gb_Y_pred)
gb_r2 = r2_score(Y_test, gb_Y_pred)

print('Gradient Boosting Mean Squared Error (MSE): %.3f' % gb_mse)
print('Gradient Boosting R-squared (R^2): %.3f' % gb_r2)
