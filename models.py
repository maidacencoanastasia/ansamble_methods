from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import concurrent.futures
import pandas as pd
import numpy as np

# Load your dataset
data = pd.read_csv("test_dataset.csv")

# Separate features (X) and target variable (Y)
X = data.drop("hemoglobin", axis=1)  # Features should exclude the target variable
Y = data["hemoglobin"]  # Target variable

# split dataset into train and test sets
X_train_full, X_test, Y_train_full, Y_test = train_test_split(X, Y, test_size=0.5, random_state=1)

# split training set into train and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train_full, Y_train_full, test_size=0.33, random_state=1)

# summarize data split
print('Train: %s, Val: %s, Test: %s' % (X_train.shape, X_val.shape, X_test.shape))

# list of base models for regression
def get_models():
    models = list()
    models.append(("linear", LinearRegression()))
    models.append(("knn", KNeighborsRegressor()))
    models.append(('cart', DecisionTreeRegressor()))
    models.append(('svm', SVR()))
    return models

# fit a base model and return its prediction
def fit_and_predict_model(name, model, X_train, Y_train, X_val):
    model.fit(X_train, Y_train)
    yhat = model.predict(X_val)
    yhat = yhat.reshape(len(yhat), 1)
    return name, yhat

def fit_ensemble(models, X_train, X_val, Y_train, Y_val):
    meta_x = list()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Train and predict each base model in parallel
        futures = {executor.submit(fit_and_predict_model, name, model, X_train, Y_train, X_val): (name, model) for name, model in models}

        for future in concurrent.futures.as_completed(futures):
            name, yhat = future.result()
            meta_x.append(yhat)

    # Stack arrays in sequence horizontally (column-wise).
    meta_x = np.hstack(meta_x)
    blender = LinearRegression()
    blender.fit(meta_x, Y_val)
    return blender

# predict using the blending ensemble for regression
# predict using the blending ensemble for regression
def predict_ensemble(models, blender, X_test):
    meta_X = list()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Predict using each base model in parallel
        futures = {executor.submit(model.predict, X_test): model for _, model in models}

        for future in concurrent.futures.as_completed(futures):
            yhat = future.result()
            yhat = yhat.reshape(len(yhat), 1)
            meta_X.append(yhat)

    meta_X = np.hstack(meta_X)
    return blender.predict(meta_X)

models = get_models()

blender = fit_ensemble(models, X_train, X_val, Y_train, Y_val)

yhat = predict_ensemble(models, blender, X_test)

# Custom accuracy metric (example: within a certain range)
threshold = 10  # Define a threshold for "accuracy"
correct_predictions = np.abs(Y_test - yhat) <= threshold
custom_accuracy = np.mean(correct_predictions) * 100

# evaluate predictions using regression metrics
mse = mean_squared_error(Y_test, yhat)
r2 = r2_score(Y_test, yhat)

print('Mean Squared Error (MSE): %.3f' % mse)
print('R-squared (R^2): %.3f' % r2)
print('Custom Accuracy (within %d units): %.3f%%' % (threshold, custom_accuracy))
