## InventoryPrediction – Stacking-based Next-Invoice Forecasting

### What this project does
- **Goal**: Predict the time until the next invoice (in days) for each hospital and forecast the next 1–4 invoice dates.
- **Approach**: A manual, time-series-safe stacking regressor that learns the mapping from the previous invoice’s features to the next invoice’s interval.

### Data inputs and preprocessing
- **Source file**: `Innovative_BU_erbitux_sales_2023_to_2025.xlsx` (loaded in `main_innovative_BU_erbitux_sales_HA.py`).
- **Filters** (in `main_innovative_BU_erbitux_sales_HA.py`):
  - `Sub Channel == 'HA Hospital'`
  - `Brand Detail == "ERBITUX 5MG/ML INJ 20ML 1'S"`
  - `AmountInHKD > 0`
- **Per-hospital prep**:
  - Sort by `InvoiceDate` and compute `Time After Last Invoices` as the day difference between consecutive `InvoiceDate`s.
  - A grouped version by `['SaleOrderNo', 'InvoiceDate', 'SoldToCustomerName', 'Brand Detail']` is built; sums are taken for `Counting Unit`, `AmountInHKD`, `Std Counting Unit`, and `Time After Last Invoices`.

### Feature engineering and target alignment
- We predict the interval to the next invoice using the **previous invoice’s features**:
  - From each `InvoiceDate`, derive `DayOfYear`, `Month`, `DayOfWeek`, and include `Counting Unit` (renamed internally to `Counting_Unit`).
  - Shift the feature matrix by 1 to align: features at time t−1 predict the interval observed at time t.
  - Drop the first row (NaNs from shifting). This prevents leakage and aligns features to the correct target.

### Model architecture (manual stacking)
- **Base learners**: `RandomForestRegressor` and `LGBMRegressor` (LightGBM).
- **Meta-learner**: `Ridge` regression.
- **TimeSeriesSplit**:
  - Adaptive number of splits based on sample size: `num_splits = min(5, max(2, n_samples // 4))`.
  - For each split, train base models on the past and produce **out-of-fold (OOF)** predictions on the next fold.
  - Train the `Ridge` meta-model on the OOF predictions.
  - If OOF rows are insufficient (very small data), fallback to training meta on in-sample base predictions from the holdout train split.

### Evaluation
- Use the **last TimeSeriesSplit split** as a holdout set.
- Metrics printed:
  - **MSE**, **RMSE**, **R²**
- A comparison frame is built with `Actual` vs `Predicted (Stacking)` aligned on the correct dates.

### Forecasting logic (multi-step)
- After evaluation, base learners are refit on all available aligned data.
- Forecasting uses the **previous-invoice feature mapping**:
  - Start from the last actual `InvoiceDate` and the last known `Counting Unit`.
  - For step k, build features from the current simulated date (day-of-year, month, day-of-week, counting unit), predict the next interval, and advance the date by that many days. Repeat iteratively.
- **Horizon**: 4 steps (i.e., forecast the next 4 invoice dates) by default.
- **Safeguards**:
  - If the predicted interval is non-finite or ≤ 0, fall back to the median of positive historical intervals; if none, use 30 days.
  - Cap extreme values: if enough history, cap at roughly `min(max(95th percentile × 2), 365)`; otherwise cap at `365` days.

### Outputs
- Aggregated text output: controlled by `AGGREGATED_OUTPUT_FILE` in `ML_model_function2.py`.
  - Default: `prediction_output_ALL.txt`.
  - Content includes metrics, last actual invoice date, the next predicted interval/date, and a list of the **next 4 forecasted invoice dates** like:
    - `1) +N days -> YYYY-MM-DD`
    - `2) +M days -> YYYY-MM-DD`
- Note: `main_innovative_BU_erbitux_sales_HA.py` currently deletes `prediction_output_ALL.txt` before runs. If you changed `AGGREGATED_OUTPUT_FILE`, update the deletion step accordingly, or delete both files before runs.

### How to run
- Ensure Python 3.9+ and install dependencies:
  ```bash
  pip install pandas numpy scikit-learn lightgbm matplotlib seaborn plotly openpyxl
  ```
- Run the HA workflow:
  ```bash
  python -u main_innovative_BU_erbitux_sales_HA.py
  ```
- Output: check `prediction_output_ALL.txt` for the aggregated results.

### Customization
- **Forecast horizon**: In `ML_model_function2.py`, change `horizon = 4` to any positive integer.
- **Aggregated output file**: In `ML_model_function2.py`, set `AGGREGATED_OUTPUT_FILE` to your preferred filename.
- **Hospitals processed**:
  - The provided `main_innovative_BU_erbitux_sales_HA.py` example runs the stacking model on one hospital table by default (the last in the filtered list). To process all hospitals, call `top_sales_hospital_prediction_stacking(...)` inside a loop over the prepared per-hospital tables.
- **Email-friendly format**:
  - The code writes text output. For sharing with non-technical stakeholders, consider exporting a summary to **Excel** (recommended). A suggested schema: `Hospital`, `LastInvoiceDate`, `PredictedIntervalDays`, `ForecastedNextInvoiceDate`, `RMSE`, `R²`.

### Key files
- `main_innovative_BU_erbitux_sales_HA.py`: Loads data, filters for HA + ERBITUX, prepares per-hospital tables, and invokes the stacking predictor.
- `ML_model_function2.py`: Implements the stacking model, time-series-safe training, multi-step (4-step) forecasting, and output writing.
- `ML_model_function.py`: Alternate Random Forest example (single-step) retained for reference.

### Notes and limitations
- Small datasets can be noisy. The model adds fallbacks and caps to avoid non-sensical forecasts (e.g., negative intervals).
- Future feature values (e.g., `Counting Unit`) are assumed to remain at the last observed level during forecasting. If you expect changes, consider modeling or providing a projected `Counting Unit` sequence.
- LightGBM may require additional system libraries on some platforms.

### Troubleshooting
- If no output is produced, confirm the input Excel is present and matches expected columns.
- If you changed `AGGREGATED_OUTPUT_FILE`, ensure the deletion step in `main_innovative_BU_erbitux_sales_HA.py` matches, or manually delete old outputs.
- For very small series (≤ ~8 rows), expect wider uncertainty; consider aggregating more history.
