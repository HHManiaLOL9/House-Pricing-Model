import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import category_encoders as ce

df = pd.read_csv(r'C:\Users\harsh\OneDrive\Desktop\python work\house_price.csv')
df = df.drop(['date', 'street', 'country'], axis=1, errors='ignore')
df = df[(df['price'] < 1_500_000) & (df['sqft_living'] < 4000)]
df = df.fillna(df.median(numeric_only=True))

df['house_age'] = 2025 - df['yr_built']
df['renovated'] = (df['yr_renovated'] > 0).astype(int)
df['sqft_ratio'] = df['sqft_living'] / (df['sqft_lot'] + 1)
df['bath_bed_ratio'] = df['bathrooms'] / (df['bedrooms'] + 1)
df['log_sqft_living'] = np.log1p(df['sqft_living'])
df['log_sqft_lot'] = np.log1p(df['sqft_lot'])
df['price_per_sqft'] = df['price'] / (df['sqft_living'] + 1)
df['total_rooms'] = df['bedrooms'] + df['bathrooms']
df['above_ratio'] = df['sqft_above'] / (df['sqft_living'] + 1)
df['density'] = df['sqft_lot'] / (df['floors'] + 1)

y = np.log1p(df['price'])
X = df.drop('price', axis=1)

categorical_cols = ['city', 'statezip']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

encoder = ce.TargetEncoder(cols=categorical_cols)
X_encoded = encoder.fit_transform(X, y)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols)
])

xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.9,
                   colsample_bytree=0.8, random_state=42, verbosity=0)
lgbm = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.9,
                     colsample_bytree=0.8, random_state=42)
cat = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, verbose=0, random_state=42)

stacked_model = StackingRegressor(
    estimators=[
        ('xgb', xgb),
        ('lgbm', lgbm),
        ('cat', cat)
    ],
    final_estimator=RidgeCV(),
    n_jobs=-1
)


pipeline = Pipeline([
    ('pre', preprocessor),
    ('model', stacked_model)
])

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


pipeline.fit(X_train, y_train)

def evaluate(title, model, X_eval, y_eval):
    preds_log = model.predict(X_eval)
    preds = np.expm1(preds_log)
    actuals = np.expm1(y_eval)
    r2 = r2_score(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae = mean_absolute_error(actuals, preds)
    print(f"{title} - R²: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    return preds, actuals

print("\n Evaluation Results (Train vs Test):\n" + "="*40)
print("  TRAINING SET RESULTS")
train_preds, train_actuals = evaluate("Train Set", pipeline, X_train, y_train)

print("\n  TEST SET RESULTS")
test_preds, test_actuals = evaluate("Test Set", pipeline, X_test, y_test)

def plot_preds(preds, actuals, title):
    plt.figure(figsize=(8,6))
    plt.scatter(actuals, preds, alpha=0.6)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"{title} - Actual vs Predicted")
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--')
    plt.show()

    plt.figure(figsize=(8,6))
    sns.histplot(actuals - preds, bins=50, kde=True)
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel("Residuals")
    plt.title(f"{title} - Residuals")
    plt.show()

plot_preds(test_preds, test_actuals, "Stacked Ensemble (Test Set)")

joblib.dump(pipeline, 'stacked_house_price_model.pkl')
print(" Model saved as 'stacked_house_price_model.pkl'")