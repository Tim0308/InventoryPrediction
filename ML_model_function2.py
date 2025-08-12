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
import os

# Aggregate output filename for all hospitals
AGGREGATED_OUTPUT_FILE = "prediction_output_ALL_Erbitux_HA.txt"


def top_sales_hospital_prediction_stacking(sales_table, horizon_days):
  hospital_name = sales_table['SoldToCustomerName'].iloc[0]
  print(f'Processing: {hospital_name}')
  # Ensure sorted by date and build features aligned to predict next interval from previous invoice features
  df_model = sales_table.sort_values(by='InvoiceDate').reset_index(drop=True).copy()

  # Feature Engineering on current invoice date
  df_model['DayOfYear'] = df_model['InvoiceDate'].dt.dayofyear
  df_model['Month'] = df_model['InvoiceDate'].dt.month
  df_model['DayOfWeek'] = df_model['InvoiceDate'].dt.dayofweek

  # Prepare features based on current invoice, then shift by 1 to use previous invoice's features
  features_current = df_model[['DayOfYear', 'Month', 'DayOfWeek', 'Counting Unit']].copy()
  features_current.rename(columns={'Counting Unit': 'Counting_Unit'}, inplace=True)
  X = features_current.shift(1)
  y = df_model['Time After Last Invoices']

  # Drop rows that became NaN due to shift and ensure target exists
  valid_mask = X.notna().all(axis=1) & y.notna()
  X = X.loc[valid_mask].reset_index(drop=True)
  y = y.loc[valid_mask].reset_index(drop=True)
  dates_aligned = df_model.loc[valid_mask, 'InvoiceDate'].reset_index(drop=True)

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

  # Fit base learners on all available data for forecasting
  rf_base.fit(X, y)
  lgbm_base.fit(X, y)

  # Helper to compute features from a date and assumed counting unit
  def build_features_from_date(input_date, counting_unit_value):
      return pd.DataFrame({
          'DayOfYear': [input_date.timetuple().tm_yday],
          'Month': [input_date.month],
          'DayOfWeek': [input_date.weekday()],
          'Counting_Unit': [counting_unit_value]
      })

  # Iterative multi-step forecast for next 4 invoices, using previous-invoice features
  last_invoice_date = df_model['InvoiceDate'].iloc[-1]
  assumed_counting_unit = df_model['Counting Unit'].iloc[-1] if 'Counting Unit' in df_model.columns else 0
  horizon = horizon_days
  forecast_intervals = []
  forecast_dates = []
  current_date = last_invoice_date
  for _ in range(horizon):
      feat_prev = build_features_from_date(current_date, assumed_counting_unit)
      base_last = np.column_stack([
          rf_base.predict(feat_prev),
          lgbm_base.predict(feat_prev)
      ])
      raw_pred = meta_model.predict(base_last)[0]
      predicted_time_interval_stack = int(np.ceil(raw_pred))

      # Enforce non-negative, realistic forecast with a robust fallback for small/noisy data
      positive_intervals_hist = y[y > 0]
      if predicted_time_interval_stack <= 0 or not np.isfinite(predicted_time_interval_stack):
          fallback_value = int(np.ceil(positive_intervals_hist.median())) if len(positive_intervals_hist) > 0 else 30
          predicted_time_interval_stack = max(1, fallback_value)
      else:
          # Optional upper cap based on historical distribution
          if len(positive_intervals_hist) >= 5:
              cap_days = int(np.ceil(np.percentile(positive_intervals_hist, 95) * 2))
              cap_days = max(7, min(cap_days, 365))
              if predicted_time_interval_stack > cap_days:
                  predicted_time_interval_stack = cap_days
          else:
              if predicted_time_interval_stack > 365:
                  predicted_time_interval_stack = 365

      next_date = current_date + pd.to_timedelta(predicted_time_interval_stack, unit='D')
      forecast_intervals.append(predicted_time_interval_stack)
      forecast_dates.append(next_date)
      current_date = next_date

  # Report first step (backward compatibility) and the 4-step outlook
  print(f"\nPredicted next interval (Stacking): {forecast_intervals[0]} days")
  print(f"Forecasted Next Invoice Date (Stacking): {forecast_dates[0].strftime('%Y-%m-%d')}")
  print("Next 4 forecasted invoice dates:")
  for idx, (ival, dte) in enumerate(zip(forecast_intervals, forecast_dates), start=1):
      print(f"  {idx}) +{ival} days -> {dte.strftime('%Y-%m-%d')}")

  # --- Prepare data for plotting ---
  results_compare = pd.DataFrame({
      'Actual': y_te_hold,
      'Predicted (Stacking)': y_pred_stack
  })
  results_compare['Date'] = dates_aligned.iloc[test_indices].values
  results_melted = results_compare.melt(id_vars='Date', var_name='Type', value_name='Time After Last Invoices')

  forecast_data = pd.DataFrame({
      'Date': [forecast_dates[0]],
      'Time After Last Invoices': [forecast_intervals[0]],
      'Type': ['Forecasted Next Invoice Date (Stacking)']
  })
  results_melted_with_forecast = pd.concat([results_melted, forecast_data], ignore_index=True)
  
  # --- Create and save forecast plot (no display) ---
  try:
      os.makedirs("result_plot", exist_ok=True)
      # Ensure chronological order for plotting
      df_plot = results_melted_with_forecast.sort_values("Date").reset_index(drop=True)
      fig_forecast = px.line(
          df_plot,
          x="Date",
          y="Time After Last Invoices",
          color="Type",
          title=f"Actual vs Predicted Time Between Invoices and Forecasted Next Date (Stacking) for {hospital_name}",
          labels={"Date": "Invoice Date", "Time After Last Invoices": "Time Between Invoices (Days)"},
      )
      fig_forecast.update_traces(mode="lines+markers", selector=dict(type="scatter"))

      safe_hospital = re.sub(r'[\\/*?:"<>|]', '_', hospital_name)
      png_path = os.path.join("result_plot", f"{safe_hospital}__stacking_forecast.png")

      # Save PNG (requires kaleido). If unavailable, fall back to Matplotlib.
      try:
          fig_forecast.write_image(png_path, format="png", scale=2, width=1200, height=600)
      except Exception:
          try:
              # Matplotlib fallback
              import matplotlib.dates as mdates
              fig, ax = plt.subplots(figsize=(12, 6))
              for plot_type, df_sub in df_plot.groupby('Type'):
                  ax.plot(df_sub['Date'], df_sub['Time After Last Invoices'], marker='o', label=plot_type)
              ax.set_title(f"Actual vs Predicted Time Between Invoices and Forecasted Next Date (Stacking) for {hospital_name}")
              ax.set_xlabel("Invoice Date")
              ax.set_ylabel("Time Between Invoices (Days)")
              ax.legend()
              ax.grid(True, alpha=0.3)
              ax.xaxis.set_major_locator(mdates.AutoDateLocator())
              ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
              fig.autofmt_xdate()
              fig.tight_layout()
              fig.savefig(png_path, dpi=200)
              plt.close(fig)
          except Exception as e2:
              print(f"Warning: PNG not saved for {hospital_name}. Install kaleido: python -m pip install -U kaleido. Error: {str(e2)[:120]}")
  except Exception:
      # Never let plotting failure break forecasting/output
      pass

  # --- Collect all results and write to file at the end ---
  output_lines = [
      f"Hospital: {hospital_name}",
      "\n--- Stacking Regressor Evaluation ---",
      f"Mean Squared Error: {mse_stack:.2f}",
      f"Root Mean Squared Error: {rmse_stack:.2f}",
      f"R-squared: {r2_stack:.2f}",
      "\n--- Forecast ---",
      f"Last Invoice Date: {last_invoice_date.strftime('%Y-%m-%d')}",
      f"Predicted time interval (Stacking): {forecast_intervals[0]} days",
      f"Forecasted Next Invoice Date (Stacking): {forecast_dates[0].strftime('%Y-%m-%d')}",
      "Next 4 forecasted invoice dates:"
  ]
  for idx, (ival, dte) in enumerate(zip(forecast_intervals, forecast_dates), start=1):
      output_lines.append(f"  {idx}) +{ival} days -> {dte.strftime('%Y-%m-%d')}")

  filename = AGGREGATED_OUTPUT_FILE
  with open(filename, "a", encoding="utf-8") as f:
      for line in output_lines:
          f.write(line + "\n")
      f.write("============================================================================= \n")
  
  
  return results_melted_with_forecast
