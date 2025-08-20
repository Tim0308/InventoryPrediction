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
from ML_model_function import top_sales_hospital_prediction
from ML_model_function2 import top_sales_hospital_prediction_stacking
from report_export import create_excel_from_prediction_summary
import os
from main_innovative_BU_ERBITUX_5MG_sales_HA import ERBITUX_sales_prediction
from main_innovative_BU_MAVENCLAD_10MG_sales_HA import MAVENCLAD_sales_prediction
from main_innovative_BU_BAVENCIO_200MG_sales_HA import BAVENCIO_sales_prediction
from main_innovative_BU_TEPMETKO_TAB_225MG_sales_HA import TEPMETKO_sales_prediction
import time

print('hello world')
excel_file_path = 'SalesM2_records_copy.xlsx'
columns_to_keep = [
    'Year', 'Quarter', 'Month', 'Original BU', 'SoldToCustomerName', 'Brand Detail', 'Std Counting Unit',
    'InvoiceDate', 'SaleOrderNo', 'Counting Unit', 'AmountInHKD', 'Std/ Bonus...', 'Sales Rep', 'Sub Channel'
]
# Replace 'Sheet1' with the name or index of the sheet you want to read
print("====== start reading excel ======")
df = pd.read_excel(excel_file_path, sheet_name='Sales_M2')
SalesM2_2023_to_2025 = df[df['Year'] >= 2023][columns_to_keep]

SalesM2_2023_to_2025_cleaned_Innovative_BU = SalesM2_2023_to_2025[(SalesM2_2023_to_2025['Original BU'] == "Innovative Medicine") 
                                                                & (SalesM2_2023_to_2025['AmountInHKD'] > 0)]

# output_path_cleaned = 'SalesM2_2023_to_2025_cleaned_Innovative_BU.xlsx'

# Display the first few rows of the new DataFrame
print(SalesM2_2023_to_2025_cleaned_Innovative_BU)

# Remove existing predictions file to avoid permission errors
predictions_file = 'predictions_summary.xlsx'
if os.path.exists(predictions_file):
    try:
        os.remove(predictions_file)
        print(f"Removed existing {predictions_file}")
    except PermissionError:
        print(f"Warning: Could not remove {predictions_file}. Please close Excel and try again.")

#  ["ERBITUX 5MG/ML INJ 20ML 1'S", 'BAVENCIO 200MG (20MG/ML) (1) - HKG', 'TEPMETKO TAB 225 MG - (60) HKG', "REBIF MULTIDOSE SYR 66MCG 4'S+13 NEEDLES", "MAVENCLAD TABS 10MG 1'S"]
ERBITUX_prediction_summary_list = ERBITUX_sales_prediction(SalesM2_2023_to_2025_cleaned_Innovative_BU)
print("================================== DONE =========================================")
# time.sleep(5)
# MAVENCLAD_sales_prediction(SalesM2_2023_to_2025_cleaned_Innovative_BU)
# print("================================== DONE =========================================")
time.sleep(5)
BAVENCIO_prediction_summary_list = BAVENCIO_sales_prediction(SalesM2_2023_to_2025_cleaned_Innovative_BU)
print("================================== DONE =========================================")
time.sleep(5)

TEPMETKO_prediction_summary_list = TEPMETKO_sales_prediction(SalesM2_2023_to_2025_cleaned_Innovative_BU)
print("================================== DONE =========================================")
time.sleep(5)

Final_prediction_summary_list = ERBITUX_prediction_summary_list + BAVENCIO_prediction_summary_list + TEPMETKO_prediction_summary_list
create_excel_from_prediction_summary(Final_prediction_summary_list, output_path='result_excel/predictions_summary.xlsx')