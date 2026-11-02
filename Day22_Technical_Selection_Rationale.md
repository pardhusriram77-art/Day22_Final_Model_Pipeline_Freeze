# Day 22 — Final Model / Pipeline Freeze

## Frozen final candidate
**pH-only Linear Regression** with median imputation, forecasting the next row's pH from the current row's pH.

## Selection rationale
The fixed chronological validation comparison included linear regression, pH-only Random Forest, full 9-feature Random Forest, a mean baseline, and a persistence baseline (next pH predicted as current pH).
- Linear Regression validation MAE: **1.2598**, RMSE: **1.5782**.
- pH-only Random Forest validation MAE: **1.2821**, RMSE: **1.5940**.
- Full-feature Random Forest validation MAE: **1.2944**, RMSE: **1.6170**.
- Persistence validation MAE: **1.6005**, RMSE: **2.0783**.

Linear Regression had the lowest validation RMSE among the compared candidates and is simpler than the Random Forest. It is therefore selected as the frozen candidate. This choice prioritizes validation performance and simplicity; it does not imply that a single pH feature is adequate for real underwater monitoring.

## Frozen data and preprocessing protocol
1. Load the supplied synthetic temporal CSV.
2. Parse `timestamp` using `pandas.to_datetime(errors='coerce')`.
3. Drop rows with invalid timestamps.
4. Stable-sort ascending by timestamp.
5. Create `target_next_ph = ph.shift(-1)` and drop rows without this target.
6. Use only current-row `ph` as the predictor.
7. Median imputation is fitted inside the scikit-learn Pipeline on development data.
8. No scaling and no random shuffling.

## Frozen split
Chronological split: earliest 70% train, next 15% validation, latest 15% test.
- Train: 1939 rows
- Validation: 416 rows
- Reserved test: 416 rows

Candidates are compared on training/validation only. After selection, Linear Regression is refit on train + validation, then evaluated once on the untouched test partition.

## Model and pipeline
`Pipeline([SimpleImputer(strategy='median'), LinearRegression()])`

## Reserved test result
- MAE: **1.4815**
- RMSE: **1.9005**
- Rows: 416

Test metrics are reported after selection and are not used to change the model.

## Reproducibility
- `Day22_Final_Linear_Regression_Pipeline.joblib` — fitted pipeline
- `Day22_Model_Metadata.json` — frozen configuration, metrics, environment and limitations
- `Day22_Validation_Comparison.csv` — candidate comparison
- `Day22_Validation_Window_Stability.csv` — three-window validation stability
- `Day22_Chronological_Split_Manifest.csv` — split boundaries
- `Day22_Reserved_Test_Predictions.csv` — held-out predictions and absolute errors
- `Day22_Reproduce.py` and `Day22_Final_Model_Reproduction.ipynb` — reproducibility code
- `water_potability_temporal_synthetic.csv` — exact dataset used

## Limitations
The timestamps were synthetically generated; the original water-potability dataset is static. The next-row pH target is consequently an exploratory pseudo-temporal target, not a true future underwater sensor measurement. The dataset has no dissolved-oxygen or pressure columns and no labelled anomaly ground truth. This frozen pipeline is suitable for reproducible academic demonstration only and must be revalidated on genuine timestamped underwater data before operational use.
