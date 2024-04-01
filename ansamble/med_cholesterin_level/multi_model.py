import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset (replace 'your_dataset.csv' with your actual dataset)
data = pd.read_csv("test_dataset.csv")

# Separate features (X) and target variable (hemoglobin)
X = data.drop("hemoglobin", axis=1)  # Features (exclude hemoglobin)
Y = data["hemoglobin"]  # Target variable

# Split the dataset into training and testing sets (e.g., 80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Create a Ridge regression model
ridge_model = Ridge(alpha=1.0)

# Train the Ridge regression model
ridge_model.fit(X_train, Y_train)

# Make predictions on the test data
ridge_Y_pred = ridge_model.predict(X_test)

# Evaluate the Ridge regression model using regression metrics
ridge_mse = mean_squared_error(Y_test, ridge_Y_pred)
ridge_r2 = r2_score(Y_test, ridge_Y_pred)

print('Ridge Regression Mean Squared Error (MSE): %.3f' % ridge_mse)
print('Ridge Regression R-squared (R^2): %.3f' % ridge_r2)

# Create a KNeighborsRegressor model
knn_model = KNeighborsRegressor(n_neighbors=5)

# Train the KNeighborsRegressor model
knn_model.fit(X_train, Y_train)

# Make predictions on the test data
knn_Y_pred = knn_model.predict(X_test)

# Evaluate the KNeighborsRegressor model using regression metrics
knn_mse = mean_squared_error(Y_test, knn_Y_pred)
knn_r2 = r2_score(Y_test, knn_Y_pred)

print('KNeighborsRegressor Mean Squared Error (MSE): %.3f' % knn_mse)
print('KNeighborsRegressor R-squared (R^2): %.3f' % knn_r2)

# Create a LinearRegression model
linear_model = LinearRegression()

# Train the LinearRegression model
linear_model.fit(X_train, Y_train)

# Make predictions on the test data
linear_Y_pred = linear_model.predict(X_test)

# Evaluate the LinearRegression model using regression metrics
linear_mse = mean_squared_error(Y_test, linear_Y_pred)
linear_r2 = r2_score(Y_test, linear_Y_pred)

print('Linear Regression Mean Squared Error (MSE): %.3f' % linear_mse)
print('Linear Regression R-squared (R^2): %.3f' % linear_r2)

# Create a DecisionTreeRegressor model
tree_model = DecisionTreeRegressor(random_state=1)

# Train the DecisionTreeRegressor model
tree_model.fit(X_train, Y_train)

# Make predictions on the test data
tree_Y_pred = tree_model.predict(X_test)

# Evaluate the DecisionTreeRegressor model using regression metrics
tree_mse = mean_squared_error(Y_test, tree_Y_pred)
tree_r2 = r2_score(Y_test, tree_Y_pred)

print('Decision Tree Regression Mean Squared Error (MSE): %.3f' % tree_mse)
print('Decision Tree Regression R-squared (R^2): %.3f' % tree_r2)
