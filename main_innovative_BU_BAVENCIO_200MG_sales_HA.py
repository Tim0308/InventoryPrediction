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



print('hello world')

# Ensure fresh aggregated output per run
aggregated_output = "prediction_output_ALL.txt"
if os.path.exists(aggregated_output):
    try:
        os.remove(aggregated_output)
    except OSError:
        pass


# # Load Excel data
# excel_file_path = 'SalesM2_records_copy.xlsx'
# sheet_name = 'Sales_M2'
# columns_to_keep = [
#     'Year', 'Quarter', 'Month', 'Original BU', 'SoldToCustomerName', 'Brand Detail', 'Std Counting Unit',
#     'InvoiceDate', 'SaleOrderNo', 'Counting Unit', 'AmountInHKD', 'Std/ Bonus...', 'Sales Rep', 'Sub Channel'
# ]

# df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
# # Filter for years >= 2023 and select relevant columns
# filtered_df = df[df['Year'] >= 2023][columns_to_keep]
# # Filter for Innovative Medicine BU and BAVENCIO brand
# Innovative_BU_BAVENCIO_sales_2023_to_2025 = filtered_df[
#     (filtered_df['Original BU'] == 'Innovative Medicine') &
#     (filtered_df['Brand Detail'] == 'BAVENCIO 5MG/ML INJ 20ML 1\'S')
# ]

# print(Innovative_BU_BAVENCIO_sales_2023_to_2025)

output_path_cleaned = 'Innovative_BU_BAVENCIO_sales_2023_to_2025.xlsx'
# Innovative_BU_BAVENCIO_sales_2023_to_2025.to_excel(output_path_cleaned, index=True)
# print(f"Cleaned data saved to: {output_path_cleaned}")

"""
===============================================================================
"""
def BAVENCIO_sales_prediction():
    # Replace 'Sheet1' with the name or index of the sheet you want to read
    Innovative_BU_BAVENCIO_sales_2023_to_2025 = pd.read_excel(output_path_cleaned)

    Innovative_BU_BAVENCIO_sales_2023_to_2025 = Innovative_BU_BAVENCIO_sales_2023_to_2025[(Innovative_BU_BAVENCIO_sales_2023_to_2025['AmountInHKD'] > 0)]


    # Display the first few rows of the DataFrame
    # print(Innovative_BU_BAVENCIO_sales_2023_to_2025)


    Innovative_BU_BAVENCIO_sales_2023_to_2025_HA = Innovative_BU_BAVENCIO_sales_2023_to_2025[
        Innovative_BU_BAVENCIO_sales_2023_to_2025['Sub Channel'] == 'HA Hospital'
    ]

    # Combine Year and Quarter for the pivot table columns
    Innovative_BU_BAVENCIO_sales_2023_to_2025_HA['YearQuarter'] = Innovative_BU_BAVENCIO_sales_2023_to_2025_HA['Year'].astype(str) + ' Q' + Innovative_BU_BAVENCIO_sales_2023_to_2025_HA['Quarter'].astype(str).str[-1]
    # print(Innovative_BU_BAVENCIO_sales_2023_to_2025_HA)

    hospital_list = Innovative_BU_BAVENCIO_sales_2023_to_2025_HA['SoldToCustomerName'].unique().tolist()
    print(f"number of hospital = {len(hospital_list)}")



    HA_hospital_list = ['PRINCESS MARGARET HOSP', 'UNITED CHRISTIAN HOSPITAL', 'TSEUNG KWAN O HOSPITAL', 
                        'QUEEN ELIZABETH HOSPITAL', 'PRINCE OF WALES HOSPITAL', 'QUEEN MARY HOSPITAL C/O PHARMACY',
                        'PAMELA YOUDE NETHERSOLE EASTERN HOSPITAL', 'TUEN MUN HOSPITAL', ]

    # top_sales_hospital = ['QUEEN ELIZABETH HOSPITAL']
    Innovative_BU_BAVENCIO_sales_2023_to_2025_HA_filtered_top_sales = Innovative_BU_BAVENCIO_sales_2023_to_2025_HA[
        (Innovative_BU_BAVENCIO_sales_2023_to_2025_HA['SoldToCustomerName'].isin(HA_hospital_list)) &
        # (Innovative_BU_BAVENCIO_sales_2023_to_2025_HA['ShipToCode From ImportData'] == '70145698') &
        (Innovative_BU_BAVENCIO_sales_2023_to_2025_HA['Brand Detail'] == 'BAVENCIO 5MG/ML INJ 20ML 1\'S')
    ]


    # Ensure InvoiceDate is datetime
    Innovative_BU_BAVENCIO_sales_2023_to_2025_HA_filtered_top_sales['InvoiceDate'] = pd.to_datetime(Innovative_BU_BAVENCIO_sales_2023_to_2025_HA_filtered_top_sales['InvoiceDate'])
    # Sort and reset index for correct diff calculation
    Innovative_BU_BAVENCIO_sales_2023_to_2025_HA_filtered_top_sales = Innovative_BU_BAVENCIO_sales_2023_to_2025_HA_filtered_top_sales.sort_values(by='InvoiceDate').reset_index(drop=True)
    # Calculate the time difference between consecutive invoices
    Innovative_BU_BAVENCIO_sales_2023_to_2025_HA_filtered_top_sales['Time After Last Invoices'] = Innovative_BU_BAVENCIO_sales_2023_to_2025_HA_filtered_top_sales['InvoiceDate'].diff().dt.days

    # print(Innovative_BU_BAVENCIO_sales_2023_to_2025_HA_filtered_top_sales)

    HA_hospital_table_list = []
    # Filter the DataFrame based on 'Brand Detail' being in the list and 'AmountInHKD' being greater than 0
    HA_hospital_list = ['PRINCESS MARGARET HOSP', 'UNITED CHRISTIAN HOSPITAL', 'TSEUNG KWAN O HOSPITAL', 
                        'QUEEN ELIZABETH HOSPITAL', 'PRINCE OF WALES HOSPITAL', 'QUEEN MARY HOSPITAL C/O PHARMACY',
                        'PAMELA YOUDE NETHERSOLE EASTERN HOSPITAL', 'TUEN MUN HOSPITAL']

    for HA_hospital in HA_hospital_list:
        df_hospital = Innovative_BU_BAVENCIO_sales_2023_to_2025_HA_filtered_top_sales[
            (Innovative_BU_BAVENCIO_sales_2023_to_2025_HA_filtered_top_sales['SoldToCustomerName'] == HA_hospital) &
            (Innovative_BU_BAVENCIO_sales_2023_to_2025_HA_filtered_top_sales['Brand Detail'] == 'BAVENCIO 5MG/ML INJ 20ML 1\'S')
        ].copy()
        # Sort and reset index for correct diff calculation
        df_hospital = df_hospital.sort_values(by='InvoiceDate').reset_index(drop=True)
        df_hospital['Time After Last Invoices'] = df_hospital['InvoiceDate'].diff().dt.days
        print(f"The number of rows in the DataFrame for {HA_hospital} is: {df_hospital.shape[0]}")
        HA_hospital_table_list.append(df_hospital)


    HA_hospital_table_list_merged_same_order = []

    for df_hospital in HA_hospital_table_list:
         # Group by 'SaleOrderNo' and sum the specified columns
        df_merged = df_hospital.groupby(['SaleOrderNo', 'InvoiceDate', 'SoldToCustomerName', 'Brand Detail']).agg(
            {'Counting Unit': 'sum', 'AmountInHKD': 'sum', 'Std Counting Unit': 'sum', 'Time After Last Invoices': 'sum'}
        ).reset_index()

        # Sort by 'InvoiceDate'
        df_merged = df_merged.sort_values(by='InvoiceDate')
        HA_hospital_table_list_merged_same_order.append(df_merged)


    # print(HA_hospital_table_list_merged_same_order[0])
    for i in range(1, len(HA_hospital_table_list_merged_same_order)):
    #   print(HA_hospital_table_list_merged_same_order[i])
        print(f"The number of rows in the merged and sorted DataFrame for {HA_hospital_list[i]} is: {HA_hospital_table_list_merged_same_order[i].shape[0]}")


    # print(HA_hospital_table_list_merged_same_order[1])
    # print(HA_hospital_table_list_merged_same_order[3].iloc[1:])

    prediction_summary_list = []

    for hospital_table in HA_hospital_table_list_merged_same_order[:2]:
        print(hospital_table.iloc[1:])
        hospital_name = hospital_table['SoldToCustomerName'].iloc[0]
        results_melted_with_forecast = top_sales_hospital_prediction_stacking(
            hospital_table.iloc[1:], 4
        )  # 4 is the horizon days (number of next invoices to predict)
        prediction_summary_list.append((hospital_name, results_melted_with_forecast))

    # Create a single Excel summary with all hospitals
    create_excel_from_prediction_summary(prediction_summary_list, output_path='predictions_summary.xlsx')