import subprocess
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import openpyxl
import re


def top_sales_hospital_prediction(sales_table):
  hospital_name = sales_table['SoldToCustomerName'].iloc[0]
  print(f'Processing: {hospital_name}')
  # print(f'prediction of {name}')
  df_model = sales_table.dropna(subset=['Time After Last Invoices']).copy()

  # Feature Engineering: Extract features from InvoiceDate
  df_model['DayOfYear'] = df_model['InvoiceDate'].dt.dayofyear
  df_model['Month'] = df_model['InvoiceDate'].dt.month
  df_model['DayOfWeek'] = df_model['InvoiceDate'].dt.dayofweek

  # display(df_model)
  # Select features (X) and target (y)
  X = df_model[['DayOfYear', 'Month', 'DayOfWeek', 'Counting Unit']]
  y = df_model['Time After Last Invoices']

  # Split the data into training and testing sets
  # We'll use a time-based split to simulate forecasting
  # The last part of the data will be used for testing
  split_index = int(len(df_model) * 0.8)  # 80% for training, 20% for testing
  X_train, X_test = X[:split_index], X[split_index:]
  y_train, y_test = y[:split_index], y[split_index:]

  # Initialize and train the Random Forest Regressor model
  rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
  rf_model.fit(X_train, y_train)

  # Make predictions on the test set
  y_pred_rf = rf_model.predict(X_test)

  # Evaluate the Random Forest model
  mse_rf = mean_squared_error(y_test, y_pred_rf)
  rmse_rf = np.sqrt(mse_rf)
  r2_rf = r2_score(y_test, y_pred_rf)

  print("\nRandom Forest Regressor Evaluation:")
  print(f"Mean Squared Error: {mse_rf:.2f}")
  print(f"Root Mean Squared Error: {rmse_rf:.2f}")
  print(f"R-squared: {r2_rf:.2f}")

  # Display predictions vs actual values for inspection
  results_df_rf = pd.DataFrame({'Actual': y_test, 'Predicted (RF)': y_pred_rf})

  # Example of predicting the time interval for the next expected invoice using Random Forest
  last_row_features_rf = X_test.iloc[-1].values.reshape(1, -1)
  predicted_time_interval_rf = rf_model.predict(last_row_features_rf)[0]

  print(f"\nPredicted time interval for the next invoice (RF, based on last test point features): {predicted_time_interval_rf:.2f} days")

  # To forecast the actual date using Random Forest predictions
  last_invoice_date_rf = df_model['InvoiceDate'].iloc[-1]
  forecasted_next_invoice_date_rf = last_invoice_date_rf + pd.to_timedelta(predicted_time_interval_rf, unit='D')

  print(f"Last Invoice Date: {last_invoice_date_rf.strftime('%Y-%m-%d')}")
  print(f"Forecasted Next Invoice Date (RF): {forecasted_next_invoice_date_rf.strftime('%Y-%m-%d')}")

  output_lines = [
        f"\nPredicted time interval for the next invoice (RF, based on last test point features): {predicted_time_interval_rf:.2f} days",
        f"Last Invoice Date: {last_invoice_date_rf.strftime('%Y-%m-%d')}",
        f"Forecasted Next Invoice Date (RF): {forecasted_next_invoice_date_rf.strftime('%Y-%m-%d')}"
    ]
  # Sanitize hospital name for filename
  safe_name = re.sub(r'[\\/*?:"<>|]', '_', hospital_name)
  filename = f"rf_prediction_output_{safe_name}.txt"
  with open(filename, "w") as f:
      for line in output_lines:
          f.write(line + "\n")
      f.write("============================================================================= \n")


  # Prepare data for comparing Linear Regression, Random Forest, and Actual
  results_compare = pd.DataFrame({
      'Actual': y_test,
      'Predicted (RF)': y_pred_rf
  })

  # Add a 'Date' column for plotting against time
  # We'll use the InvoiceDate from the test set for corresponding predictions
  results_compare['Date'] = df_model['InvoiceDate'].iloc[split_index:].values

  # Melt the DataFrame to long format for Plotly Express
  results_melted = results_compare.melt(id_vars='Date', var_name='Type', value_name='Time After Last Invoices')

 

  # To show the forecasted next date on the graph, we can add an annotation or another trace
  # For simplicity, let's just add a marker for the forecasted date
  forecast_data = pd.DataFrame({
      'Date': [forecasted_next_invoice_date_rf],
      'Time After Last Invoices': [predicted_time_interval_rf],
      'Type': ['Forecasted Next Invoice Date (RF)']
  })

  # Add the forecast point to the melted dataframe
  results_melted_with_forecast = pd.concat([results_melted, forecast_data], ignore_index=True)