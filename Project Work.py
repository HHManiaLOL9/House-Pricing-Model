import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline


# ===========================
# preprocess data
# ===========================


df = pd.read_csv('"C:\Users\harsh\Downloads\house_price.csv"')
df = df.drop(['date', 'street', 'country'], axis=1)


# Features
label_encoders = {}
for col in ['city', 'statezip']:
   le = LabelEncoder()
   df[col] = le.fit_transform(df[col])
   label_encoders[col] = le


df = df.fillna(df.median(numeric_only=True))


X = df.drop('price', axis=1)
y = df['price']


# ===========================
# Split into Train, Val, Test
# ===========================


X_train_full, X_test, y_train_full, y_test = train_test_split(
   X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
   X_train_full, y_train_full, test_size=0.2, random_state=42
)


# ===========================
# parameter
# ===========================


pipelines = {
   'Random Forest': Pipeline([
       ('scaler', StandardScaler()),
       ('model', RandomForestRegressor(random_state=42))
   ]),
   'SVR': Pipeline([
       ('scaler', StandardScaler()),
       ('model', SVR())
   ]),
   'KNN': Pipeline([
       ('scaler', StandardScaler()),
       ('model', KNeighborsRegressor())
   ])
}


param_grids = {
   'Random Forest': {'model__n_estimators': [100, 200, 300], 'model__max_depth': [None, 10, 20]},
   'SVR': {'model__C': [10, 100], 'model__epsilon': [0.1, 0.2]},
   'KNN': {'model__n_neighbors': [3, 5, 7]}
}


best_models = {}


# ===========================
# Training
# ===========================


for name, pipeline in pipelines.items():
   print(f"\nTraining and tuning {name}...")
   gscv = GridSearchCV(
       pipeline, param_grids[name], cv=3,
       scoring='neg_mean_squared_error', return_train_score=True
   )
   gscv.fit(X_train, y_train)
   best_models[name] = gscv.best_estimator_
  
   print(f"Best parameters for {name}: {gscv.best_params_}")
   results_df = pd.DataFrame(gscv.cv_results_)
   print(results_df[['params', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']])


# ===========================
# Validation and Test 
# ===========================


for name, model in best_models.items():
   for dataset_name, X_eval, y_eval in [('Validation', X_val, y_val), ('Test', X_test, y_test)]:
       preds = model.predict(X_eval)
       rmse = np.sqrt(mean_squared_error(y_eval, preds))
       mae = mean_absolute_error(y_eval, preds)
       r2 = r2_score(y_eval, preds)
       print(f"{name} on {dataset_name} set - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")


# ===========================
# Feature Graph
# ===========================


best_rf = best_models['Random Forest']
rf_model = best_rf.named_steps['model']
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]


plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices], y=X.columns[indices])
plt.title('Feature Importances from Random Forest')
plt.show()


# ===========================
# Plots
# ===========================


for name, model in best_models.items():
   preds = model.predict(X_test)
   residuals = y_test - preds


   # Scatter plot: Actual vs Predicted
   plt.figure(figsize=(8,6))
   plt.scatter(y_test, preds, alpha=0.6)
   plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
   plt.xlabel('Actual Price')
   plt.ylabel('Predicted Price')
   plt.title(f'{name} Predictions vs Actual')
   plt.show()


   # Residual plot: Predicted vs Residual
   plt.figure(figsize=(8,6))
   sns.scatterplot(x=preds, y=residuals, alpha=0.6)
   plt.axhline(0, color='red', linestyle='--')
   plt.xlabel('Predicted Price')
   plt.ylabel('Residual (Actual - Predicted)')
   plt.title(f'Residual Plot for {name}')
   plt.show()


# ===========================
# Random Forest model
# ===========================


joblib.dump(best_rf, 'best_random_forest_pipeline.pkl')
for col, le in label_encoders.items():
   joblib.dump(le, f'label_encoder_{col}.pkl')
print("Saved Random Forest pipeline and label encoders.")


# ===========================
# Prediction
# ===========================


def predict_new_house(new_data_csv):
   new_data = pd.read_csv(new_data_csv)
   for col in ['city', 'statezip']:
       le = joblib.load(f'label_encoder_{col}.pkl')
       new_data[col] = le.transform(new_data[col])
   new_data = new_data.fillna(new_data.median(numeric_only=True))
   model = joblib.load('best_random_forest_pipeline.pkl')
   predictions = model.predict(new_data)
   return predictions
