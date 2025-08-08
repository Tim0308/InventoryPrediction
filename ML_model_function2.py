import subprocess
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
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

  # Use TimeSeriesSplit for a robust, leakage-free train/test split
  n_samples = len(X)
  # Adaptive number of splits to avoid errors on small datasets
  num_splits = min(5, max(2, n_samples // 4))
  tscv = TimeSeriesSplit(n_splits=num_splits)
  all_splits = list(tscv.split(X, y))
  train_indices, test_indices = all_splits[-1]  # Use the last split for training and testing
  # Define holdout train/test sets upfront for later use
  X_tr_hold, X_te_hold = X.iloc[train_indices], X.iloc[test_indices]
  y_tr_hold, y_te_hold = y.iloc[train_indices], y.iloc[test_indices]
  
  X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
  y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

  # ----- Manual, time-series-safe stacking -----
  # Base learners
  rf_base = RandomForestRegressor(n_estimators=100, random_state=42)
  lgbm_base = LGBMRegressor(n_estimators=100, random_state=42,
                            min_data_in_leaf=1, min_data_in_bin=1)

  # Out-of-fold predictions for meta-model training (forward chaining)
  oof_preds = np.full((n_samples, 2), np.nan, dtype=float)
  for tr_idx, val_idx in tscv.split(X, y):
      X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
      X_val = X.iloc[val_idx]

      rf_base_fold = RandomForestRegressor(n_estimators=100, random_state=42)
      lgbm_base_fold = LGBMRegressor(n_estimators=100, random_state=42,
                                     min_data_in_leaf=1, min_data_in_bin=1)
      rf_base_fold.fit(X_tr, y_tr)
      lgbm_base_fold.fit(X_tr, y_tr)

      oof_preds[val_idx, 0] = rf_base_fold.predict(X_val)
      oof_preds[val_idx, 1] = lgbm_base_fold.predict(X_val)

  # Build meta training set from rows with OOF predictions
  valid_mask = ~np.isnan(oof_preds).any(axis=1)
  X_meta_train = oof_preds[valid_mask]
  y_meta_train = y.iloc[valid_mask]

  # Meta-model with regularization to reduce overfitting
  meta_model = Ridge(alpha=1.0)
  if X_meta_train.shape[0] >= 2:
      meta_model.fit(X_meta_train, y_meta_train)
  else:
      # Fallback: fit meta-model on in-sample base predictions from the holdout train split
      rf_tmp = RandomForestRegressor(n_estimators=100, random_state=42)
      lgbm_tmp = LGBMRegressor(n_estimators=100, random_state=42,
                               min_data_in_leaf=1, min_data_in_bin=1)
      rf_tmp.fit(X_tr_hold, y_tr_hold)
      lgbm_tmp.fit(X_tr_hold, y_tr_hold)
      base_pred_tr = np.column_stack([
          rf_tmp.predict(X_tr_hold),
          lgbm_tmp.predict(X_tr_hold)
      ])
      meta_model.fit(base_pred_tr, y_tr_hold)

  # Evaluate on the last split (holdout)
  rf_base.fit(X_tr_hold, y_tr_hold)
  lgbm_base.fit(X_tr_hold, y_tr_hold)
  base_pred_test = np.column_stack([
      rf_base.predict(X_te_hold),
      lgbm_base.predict(X_te_hold)
  ])
  y_pred_stack = meta_model.predict(base_pred_test)

  # Make and evaluate predictions on the holdout test split
  mse_stack = mean_squared_error(y_te_hold, y_pred_stack)
  rmse_stack = np.sqrt(mse_stack)
  r2_stack = r2_score(y_te_hold, y_pred_stack)

  print("\nStacking Regressor Evaluation:")
  print(f"Mean Squared Error: {mse_stack:.2f}")
  print(f"Root Mean Squared Error: {rmse_stack:.2f}")
  print(f"R-squared: {r2_stack:.2f}")

  # Forecast the next interval using the manual stacking model (use last observed invoice features)
  last_row_features = X.iloc[-1].values.reshape(1, -1)
  # Round prediction up to the next whole day
  # Fit base learners on all available data for forecasting
  rf_base.fit(X, y)
  lgbm_base.fit(X, y)
  base_last = np.column_stack([
      rf_base.predict(last_row_features),
      lgbm_base.predict(last_row_features)
  ])
  raw_pred = meta_model.predict(base_last)[0]
  predicted_time_interval_stack = int(np.ceil(raw_pred))
  last_invoice_date = df_model['InvoiceDate'].iloc[-1]
  forecasted_next_invoice_date_stack = last_invoice_date + pd.to_timedelta(predicted_time_interval_stack, unit='D')
  
  print(f"\nPredicted next interval (Stacking): {predicted_time_interval_stack} days")
  print(f"Forecasted Next Invoice Date (Stacking): {forecasted_next_invoice_date_stack.strftime('%Y-%m-%d')}")

  # --- Prepare data for plotting ---
  results_compare = pd.DataFrame({
      'Actual': y_te_hold,
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