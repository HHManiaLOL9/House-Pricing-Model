import ssl
import certifi


ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, StackingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib


# Step 1: Load California Housing dataset (real data)
california = fetch_california_housing()
X_raw = california.data
y_raw = california.target
df = pd.DataFrame(X_raw, columns=california.feature_names)


# Step 2: Feature Engineering
df['feature_sum'] = df.sum(axis=1)
df['feature_mean'] = df.mean(axis=1)
df['feature_std'] = df.std(axis=1)
X = df.values
y = y_raw


# Step 3: Outlier removal using Isolation Forest
iso = IsolationForest(contamination=0.01, random_state=42)
mask = iso.fit_predict(X) != -1
X, y = X[mask], y[mask]


# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 5: Define Stacking Regressor
stack = StackingRegressor(
   estimators=[
       ('xgb', XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)),
       ('rf', RandomForestRegressor(n_estimators=100)),
       ('svr', SVR(C=1.0))
   ],
   final_estimator=Ridge(alpha=1.0)
)


pipeline = Pipeline([
   ('scaler', StandardScaler()),
   ('model', stack)
])


# Step 6: Hyperparameter tuning
param_dist = {
   'model__final_estimator__alpha': [0.1, 1.0, 10.0],
   'model__xgb__n_estimators': [100, 200],
   'model__xgb__learning_rate': [0.05, 0.1],
   'model__xgb__max_depth': [3, 5, 7],
}


search = RandomizedSearchCV(
   pipeline, param_distributions=param_dist,
   n_iter=10, cv=3, scoring='r2', n_jobs=-1, random_state=42
)
search.fit(X_train, y_train)


# Step 7: Evaluation
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='r2')


print("Best Params:", search.best_params_)
print("MSE:", mse)
print("R² Score:", r2)
print("Mean CV R² Score:", cv_scores.mean())


# Step 8: Save model
joblib.dump(best_model, "stacked_regression_model.pkl")


# Step 9: Plot Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.title("Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.grid(True)
plt.show()


# Step 10: Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuals vs Predicted")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.grid(True)
plt.show()