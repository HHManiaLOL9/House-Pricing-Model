# House-Pricing-Model
House Price Prediction – Stacked Regression Model:

This project builds an ensemble regression model to predict house prices using machine learning. It combines multiple powerful algorithms (XGBoost, LightGBM, and CatBoost) through a stacked regression approach for high predictive accuracy and robustness.

Project Overview:

The goal of this project is to create a reliable model that can estimate house prices based on various property and location features. The dataset includes details like number of bedrooms, bathrooms, square footage, year built, and more.

The project involves data cleaning, feature engineering, encoding categorical variables, and building a stacked ensemble pipeline using scikit-learn.

Key Features:
Data Preprocessing

Removal of outliers and irrelevant columns

Handling of missing values

Feature scaling using StandardScaler

Target encoding for categorical features

Feature Engineering

New derived features were created to enhance the model’s ability to capture hidden relationships in the data, such as:

house_age

renovated

sqft_ratio

bath_bed_ratio

price_per_sqft

log_sqft_living

total_rooms

above_ratio

density

Stacked Ensemble Model

The final model combines:

XGBRegressor (XGBoost)

LGBMRegressor (LightGBM)

CatBoostRegressor
with a RidgeCV meta-model as the final estimator.

Evaluation Metrics

R² Score

Root Mean Squared Error (RMSE)

Mean Absolute Error (MAE)

Visualization

Scatter plot for Actual vs Predicted prices

Residuals distribution plot using Seaborn

Tech Stack

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

XGBoost, LightGBM, CatBoost

Category Encoders, Joblib

Files
File	Description
Housing.csv	Dataset containing housing information
Main Model.py	Main script for data preprocessing, model training, evaluation, and saving
stacked_house_price_model.pkl	Trained stacked regression model (generated after training)
How to Run

Clone the repository

git clone https://github.com/yourusername/house-price-regression.git
cd house-price-regression


Install dependencies

pip install -r requirements.txt


Run the main script

python "Main Model.py"


The trained model will be saved as:

stacked_house_price_model.pkl

Results

The stacked model shows strong predictive performance on both training and test data, demonstrating balanced generalization and low error metrics.

Example outputs include:

R² Score, RMSE, and MAE for both training and test sets

Actual vs Predicted plots for visual inspection

Model Output

After training, the model is serialized and saved using Joblib:

stacked_house_price_model.pkl


This file can be reloaded later for predictions without retraining.

Future Improvements

Add hyperparameter tuning via Optuna or GridSearchCV

Deploy using Flask or Streamlit

Integrate advanced feature selection techniques

Include geospatial features for improved location-based insights
