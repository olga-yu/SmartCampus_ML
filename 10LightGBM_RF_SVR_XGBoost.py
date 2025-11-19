from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline   # <<< added

# LightGBM
from lightgbm import LGBMRegressor

# Optional: Try to import XGBoost
try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False
    print("XGBoost not installed. You can install it using: pip install xgboost")

# ===================== Load dataset =====================
dataset = pd.read_csv("/content/drive/MyDrive/PHD/DATA/Nov/10full_dataset_final.csv")

# If your CSV has commas as decimal separators, convert to float
for col in ['temp', 'humidity', 'solarradiation']:
    dataset[col] = dataset[col].astype(str).str.replace(',', '.').astype(float)

# ===================== Ensure proper time ordering =====================
# Convert 'date' to datetime and sort chronologically
dataset['date'] = pd.to_datetime(dataset['date'], dayfirst=False)  # format like 9/13/2021
dataset = dataset.sort_values('date').reset_index(drop=True)

# ===================== Features / Target =====================
numeric_features = [
   'is_saturday','is_sunday','holiday','sealevelpressure',
   'tw_non_study','tw_semester2','season_Autumn',  'temp','humidity','solarradiation'
]

X = dataset[numeric_features]
y = dataset['normalised_attendance']

# ===================== Blocked 70/30 time-series split =====================
split_idx = int(len(dataset) * 0.7)

X_train = X.iloc[:split_idx]
y_train = y.iloc[:split_idx]

X_test  = X.iloc[split_idx:]
y_test  = y.iloc[split_idx:]

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# ===================== Helper: evaluation =====================
def evaluate_model(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    mae = mean_absolute_error(y_te, y_pred)
    mse = mean_squared_error(y_te, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_te, y_pred)

    print(f"\n{name} Results:")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    return y_pred

# ===================== SVR (baseline, needs scaling) =====================
scaler = StandardScaler()
X_train_svr = X_train.copy()
X_test_svr = X_test.copy()

X_train_svr[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_svr[numeric_features]  = scaler.transform(X_test[numeric_features])

svr_model = SVR(kernel='rbf', C=10, epsilon=0.1)
y_pred_svr = evaluate_model("SVR (baseline)", svr_model, X_train_svr, X_test_svr, y_train, y_test)

# ===================== Random Forest (baseline) =====================
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
y_pred_rf = evaluate_model("Random Forest (baseline)", rf_model, X_train, X_test, y_train, y_test)

# ===================== XGBoost (baseline, if available) =====================
if xgb_available:
    xgb_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
    y_pred_xgb = evaluate_model("XGBoost (baseline)", xgb_model, X_train, X_test, y_train, y_test)

# ===================== LightGBM (baseline) =====================
lgbm_model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)

y_pred_lgbm = evaluate_model("LightGBM (baseline)", lgbm_model, X_train, X_test, y_train, y_test)

# ===================== Time-series CV for GridSearch =====================
tscv = TimeSeriesSplit(n_splits=5)

# ===================== Random Forest + GridSearchCV =====================
param_grid_rf = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)

grid_search_rf = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid_rf,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

print("\nRunning GridSearchCV for Random Forest...")
grid_search_rf.fit(X_train, y_train)

print("\nBest parameters found for Random Forest:")
print(grid_search_rf.best_params_)
print(f"Best CV MAE (RF): {-grid_search_rf.best_score_:.4f}")

best_rf = grid_search_rf.best_estimator_

# Evaluate best RF on test set
y_pred_rf_best = best_rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf_best)
mse_rf = mean_squared_error(y_test, y_pred_rf_best)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf_best)

print("\nRandom Forest (Best from GridSearchCV) Results on TEST set:")
print(f"MAE:  {mae_rf:.4f}")
print(f"MSE:  {mse_rf:.4f}")
print(f"RMSE: {rmse_rf:.4f}")
print(f"R²:   {r2_rf:.4f}")

# ===================== SVR + GridSearchCV (with Pipeline & scaling) =====================
svr_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf'))
])

param_grid_svr = {
    'svr__C': [1, 10, 100],
    'svr__epsilon': [0.01, 0.1, 0.2],
    'svr__gamma': ['scale', 'auto']
}

grid_search_svr = GridSearchCV(
    estimator=svr_pipeline,
    param_grid=param_grid_svr,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

print("\nRunning GridSearchCV for SVR...")
grid_search_svr.fit(X_train, y_train)

print("\nBest parameters found for SVR:")
print(grid_search_svr.best_params_)
print(f"Best CV MAE (SVR): {-grid_search_svr.best_score_:.4f}")

best_svr = grid_search_svr.best_estimator_

# Evaluate best SVR on test set
y_pred_svr_best = best_svr.predict(X_test)
mae_svr = mean_absolute_error(y_test, y_pred_svr_best)
mse_svr = mean_squared_error(y_test, y_pred_svr_best)
rmse_svr = np.sqrt(mse_svr)
r2_svr = r2_score(y_test, y_pred_svr_best)

print("\nSVR (Best from GridSearchCV) Results on TEST set:")
print(f"MAE:  {mae_svr:.4f}")
print(f"MSE:  {mse_svr:.4f}")
print(f"RMSE: {rmse_svr:.4f}")
print(f"R²:   {r2_svr:.4f}")



# ===================== LightGBM + GridSearchCV =====================
param_grid_lgbm = {
    'num_leaves': [15, 31, 63],
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
}

grid_search_lgbm = GridSearchCV(
    estimator=LGBMRegressor(random_state=42),
    param_grid=param_grid_lgbm,
    cv=tscv,                            # time-aware CV
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

print("\nRunning GridSearchCV for LightGBM...")
grid_search_lgbm.fit(X_train, y_train)

print("\nBest parameters found for LightGBM:")
print(grid_search_lgbm.best_params_)
print(f"Best CV MAE (LGBM): {-grid_search_lgbm.best_score_:.4f}")

best_lgbm = grid_search_lgbm.best_estimator_

# Evaluate best LightGBM on test set
y_pred_lgbm_best = best_lgbm.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_lgbm_best)
mse = mean_squared_error(y_test, y_pred_lgbm_best)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_lgbm_best)

print("\nLightGBM (Best from GridSearchCV) Results on TEST set:")
print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# ===================== XGBoost + GridSearchCV (if available) =====================
if xgb_available:
    param_grid_xgb = {
        'n_estimators': [200, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    xgb_base = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )

    grid_search_xgb = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid_xgb,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )

    print("\nRunning GridSearchCV for XGBoost...")
    grid_search_xgb.fit(X_train, y_train)

    print("\nBest parameters found for XGBoost:")
    print(grid_search_xgb.best_params_)
    print(f"Best CV MAE (XGB): {-grid_search_xgb.best_score_:.4f}")

    best_xgb = grid_search_xgb.best_estimator_

    # Evaluate best XGB on test set
    y_pred_xgb_best = best_xgb.predict(X_test)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb_best)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb_best)
    rmse_xgb = np.sqrt(mse_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb_best)

    print("\nXGBoost (Best from GridSearchCV) Results on TEST set:")
    print(f"MAE:  {mae_xgb:.4f}")
    print(f"MSE:  {mse_xgb:.4f}")
    print(f"RMSE: {rmse_xgb:.4f}")
    print(f"R²:   {r2_xgb:.4f}")

# ===================== CatBoost (baseline) =====================
cat_model = CatBoostRegressor(
    random_state=42,
    loss_function='MAE',
    verbose=0
)

y_pred_cat = evaluate_model(
    "CatBoost (baseline)",
    cat_model,
    X_train,
    X_test,
    y_train,
    y_test
)

# ===================== CatBoost + GridSearchCV =====================
param_grid_cat = {
    'depth': [4, 6],
    'learning_rate': [0.03],
    'iterations': [300, 500]
}

cat_base = CatBoostRegressor(
    random_state=42,
    loss_function='MAE',
    verbose=0
)

grid_search_cat = GridSearchCV(
    estimator=cat_base,
    param_grid=param_grid_cat,
    cv=tscv,                            # time-aware CV
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

print("\nRunning GridSearchCV for CatBoost...")
grid_search_cat.fit(X_train, y_train)

print("\nBest parameters found for CatBoost:")
print(grid_search_cat.best_params_)
print(f"Best CV MAE (CatBoost): {-grid_search_cat.best_score_:.4f}")
best_cat = grid_search_cat.best_estimator_

# Predict on the test set
y_pred_cat_best = best_cat.predict(X_test)

# Calculate metrics
mae_cat = mean_absolute_error(y_test, y_pred_cat_best)
mse_cat = mean_squared_error(y_test, y_pred_cat_best)
rmse_cat = np.sqrt(mse_cat)
r2_cat = r2_score(y_test, y_pred_cat_best)

print("\nCatBoost (Best from GridSearchCV) Results on TEST set:")
print(f"MAE:  {mae_cat:.4f}")
print(f"MSE:  {mse_cat:.4f}")
print(f"RMSE: {rmse_cat:.4f}")
print(f"R²:   {r2_cat:.4f}")



from sklearn.model_selection import cross_val_score
import numpy as np

def metric_ci(model, X, y, cv, scoring):
    """
    Returns mean and 95% CI for a given metric.
    For loss-type scores (neg_...), we flip the sign.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    if scoring.startswith("neg_"):
        scores = -scores  # make them positive again

    mean = scores.mean()
    std = scores.std(ddof=1)
    n = len(scores)
    ci95 = 1.96 * std / np.sqrt(n)
    return mean, mean - ci95, mean + ci95

models = {
    "SVR (best)":       best_svr,
    "Random Forest":    best_rf,
    "LightGBM":         best_lgbm,
    "CatBoost":         best_cat,
}




rows = []

for name, model in models.items():
    mae_mean, mae_lo, mae_hi   = metric_ci(model, X_train, y_train, tscv, "neg_mean_absolute_error")
    rmse_mean, rmse_lo, rmse_hi = metric_ci(model, X_train, y_train, tscv, "neg_root_mean_squared_error")
    r2_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="r2", n_jobs=-1)
    
    # Calculate mean and CI for R2 scores
    r2_mean = r2_scores.mean()
    r2_std = r2_scores.std(ddof=1) if len(r2_scores) > 1 else 0  # Handle case with single fold or no std
    n_r2 = len(r2_scores)
    r2_ci = 1.96 * r2_std / np.sqrt(n_r2) if n_r2 > 0 else 0

    rows.append({
        "Model": name,
        "MAE_mean": mae_mean,   "MAE_low": mae_lo,   "MAE_high": mae_hi,
        "RMSE_mean": rmse_mean, "RMSE_low": rmse_lo, "RMSE_high": rmse_hi,
        "R2_mean": r2_mean,     "R2_low": r2_mean - r2_ci, "R2_high": r2_mean + r2_ci
    })

ci_table = pd.DataFrame(rows)
print(ci_table)




# ===================== Residual Analysis for Best Models =====================

import matplotlib.pyplot as plt
import numpy as np

best_model_preds = {
    "SVR (best)": y_pred_svr_best,
    "Random Forest (best)": y_pred_rf_best,
    "LightGBM (best)": y_pred_lgbm_best,
    "CatBoost (best)": y_pred_cat_best,
}

for name, y_pred in best_model_preds.items():
    residuals = y_test - y_pred

    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, linestyle='--', color='red', label='Zero residual')
    plt.xlabel("Predicted normalised attendance")
    plt.ylabel("Residuals (actual − predicted)")
    plt.title(f"Residual plot – {name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



# ===================== Plots (baseline models) =====================

# CatBoost baseline plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_cat, alpha=0.6, label='CatBoost (baseline)')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', label='Ideal Fit')
plt.xlabel('Actual Normalised Attendance')
plt.ylabel('Predicted Normalised Attendance')
plt.title('CatBoost (baseline): Actual vs Predicted')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# SVR baseline plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_svr, alpha=0.6, label='SVR (baseline)')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', label='Perfect prediction')
plt.xlabel('Actual Normalised Attendance')
plt.ylabel('Predicted Normalised Attendance')
plt.title('SVR (baseline): Actual vs Predicted')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Random Forest baseline plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.6, label='Random Forest (baseline)')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', label='Ideal Fit')
plt.xlabel('Actual Normalised Attendance')
plt.ylabel('Predicted Normalised Attendance')
plt.title('Random Forest (baseline): Actual vs Predicted')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# XGBoost baseline plot (if available)
if xgb_available:
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_xgb, alpha=0.6, label='XGBoost (baseline)')
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', label='Ideal Fit')
    plt.xlabel('Actual Normalised Attendance')
    plt.ylabel('Predicted Normalised Attendance')
    plt.title('XGBoost (baseline): Actual vs Predicted')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Best LightGBM plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lgbm_best, alpha=0.6, label='LightGBM (best)')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', label='Ideal Fit')
plt.xlabel('Actual Normalised Attendance')
plt.ylabel('Predicted Normalised Attendance')
plt.title('LightGBM (GridSearchCV Best): Actual vs Predicted')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

