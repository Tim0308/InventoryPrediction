import subprocess
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import openpyxl
import re


def top_sales_hospital_prediction_stacking(sales_table):
  hospital_name = sales_table['SoldToCustomerName'].iloc[0]
  print(f'Processing: {hospital_name}')
  df_model = sales_table.dropna(subset=['Time After Last Invoices']).copy()

  # Feature Engineering
  df_model['DayOfYear'] = df_model['InvoiceDate'].dt.dayofyear
  df_model['Month'] = df_model['InvoiceDate'].dt.month
  df_model['DayOfWeek'] = df_model['InvoiceDate'].dt.dayofweek

  # Prepare features and avoid whitespace in column names
  X = df_model[['DayOfYear', 'Month', 'DayOfWeek', 'Counting Unit']].copy()
  X.rename(columns={'Counting Unit': 'Counting_Unit'}, inplace=True)
  y = df_model['Time After Last Invoices']

  # Use TimeSeriesSplit for a robust train/test split
  tscv = TimeSeriesSplit(n_splits=5)
  all_splits = list(tscv.split(X, y))
  train_indices, test_indices = all_splits[-1] # Use the last split for training and testing
  
  X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
  y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

  # ----- Stacking Ensemble Model -----
  # Configure base learners, ensure LightGBM can handle small datasets
  estimators = [
      ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
      ('lgbm', LGBMRegressor(n_estimators=100, random_state=42,
                             min_data_in_leaf=1, min_data_in_bin=1))
  ]
  
  # The StackingRegressor will use its own internal TimeSeriesSplit for cross-validation
  # Use simple K-fold partitioning for stacking (cross_val_predict requires non-overlapping folds)
  stack_model = StackingRegressor(
      estimators=estimators,
      final_estimator=LinearRegression(),
      cv=5,
      n_jobs=-1
  )

  # Light hyperparameter tuning (small search space) to keep runtime low
  from sklearn.model_selection import RandomizedSearchCV
  param_dist = {
      'rf__n_estimators': [50,100, 200],
      'lgbm__n_estimators': [50,100, 200],
      'lgbm__min_data_in_leaf': [1, 5, 10]
  }
  tuner = RandomizedSearchCV(
      stack_model,
      param_distributions=param_dist,
      n_iter=5,
      cv=3,
      scoring='neg_mean_squared_error',
      n_jobs=-1,
      random_state=42
  )
  tuner.fit(X_train, y_train)
  best_model = tuner.best_estimator_
  print(f"Best hyperparameters: {tuner.best_params_}")
  # Train final stacking model with best parameters
  best_model.fit(X_train, y_train)

  # Make and evaluate predictions
  y_pred_stack = best_model.predict(X_test)
  mse_stack = mean_squared_error(y_test, y_pred_stack)
  rmse_stack = np.sqrt(mse_stack)
  r2_stack = r2_score(y_test, y_pred_stack)

  print("\nStacking Regressor Evaluation:")
  print(f"Mean Squared Error: {mse_stack:.2f}")
  print(f"Root Mean Squared Error: {rmse_stack:.2f}")
  print(f"R-squared: {r2_stack:.2f}")

  # Forecast the next interval using the tuned stacking model
  last_row_features = X_test.iloc[-1].values.reshape(1, -1)
  # Round prediction up to the next whole day
  raw_pred = best_model.predict(last_row_features)[0]
  predicted_time_interval_stack = int(np.ceil(raw_pred))
  last_invoice_date = df_model['InvoiceDate'].iloc[-1]
  forecasted_next_invoice_date_stack = last_invoice_date + pd.to_timedelta(predicted_time_interval_stack, unit='D')
  
  print(f"\nPredicted next interval (Stacking): {predicted_time_interval_stack} days")
  print(f"Forecasted Next Invoice Date (Stacking): {forecasted_next_invoice_date_stack.strftime('%Y-%m-%d')}")

  # --- Prepare data for plotting ---
  results_compare = pd.DataFrame({
      'Actual': y_test,
      'Predicted (Stacking)': y_pred_stack
  })
  results_compare['Date'] = df_model['InvoiceDate'].iloc[test_indices].values
  results_melted = results_compare.melt(id_vars='Date', var_name='Type', value_name='Time After Last Invoices')

  forecast_data = pd.DataFrame({
      'Date': [forecasted_next_invoice_date_stack],
      'Time After Last Invoices': [predicted_time_interval_stack],
      'Type': ['Forecasted Next Invoice Date (Stacking)']
  })
  results_melted_with_forecast = pd.concat([results_melted, forecast_data], ignore_index=True)

  # --- Collect all results and write to file at the end ---
  output_lines = [
      f"Hospital: {hospital_name}",
      "\n--- Stacking Regressor Evaluation ---",
      f"Mean Squared Error: {mse_stack:.2f}",
      f"Root Mean Squared Error: {rmse_stack:.2f}",
      f"R-squared: {r2_stack:.2f}",
      "\n--- Forecast ---",
      f"Last Invoice Date: {last_invoice_date.strftime('%Y-%m-%d')}",
      f"Predicted time interval (Stacking): {predicted_time_interval_stack} days",
      f"Forecasted Next Invoice Date (Stacking): {forecasted_next_invoice_date_stack.strftime('%Y-%m-%d')}"
  ]

  safe_name = re.sub(r'[\\/*?:"<>|]', '_', hospital_name)
  filename = f"prediction_output_{safe_name}.txt"
  with open(filename, "w") as f:
      for line in output_lines:
          f.write(line + "\n")
      f.write("============================================================================= \n")