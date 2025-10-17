#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PV Production ML – Fast version (keeps all models & charts)
- Διατηρεί RF / GB / XGB και όλα τα plots
- Ενσωματώνει Capacity από ERMIS
- Γρήγορο feature engineering, μικρότερο CV, parallel fit
"""

from __future__ import annotations
from datetime import datetime, time, timedelta
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

# --- EnergyQuantified ---
from energyquantified import EnergyQuantified
from energyquantified.time import Frequency

# --- ML / Plots ---
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error, mean_absolute_error
from xgboost import XGBRegressor
from inspect import signature

# =========================
# 1) ΡΥΘΜΙΣΕΙΣ / CONFIG
# =========================
EQ_API_KEY = "9b1b06b8-0074f355-02550874-2703fde3"   # <- βάλε δικό σου
ERMIS_XLSX = "ERMIS_II - Real1Q.xlsx"                # Sheet1: Date,Hour,Production,...,Capacity
ERMIS_SHEET = "Sheet1"

EQ_FREQUENCY = Frequency.PT15M  # PT15M για 15λεπτα

PV_SOLAR_DAYLIGHT_THRESHOLD = 5.0
EPSILON = 1e-6
DAY_WEIGHT = 3.0

RANDOM_STATE = 42

# =========================
# 2) EQ FORECASTS (15')
# =========================
def fetch_eq_forecasts_15m(api_key: str) -> pd.DataFrame:
    eq = EnergyQuantified(api_key=api_key)

    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)
    begin = datetime.combine(today, time(0, 0))
    end = datetime.combine(tomorrow + timedelta(days=1), time(0, 0))

    def load(series_name: str, col_name: str) -> pd.DataFrame:
        frames = []
        for da in (0, 1):
            inst = eq.instances.relative(
                series_name, begin=begin, end=end,
                tag='ec', days_ahead=da, time_of_day=time(0, 0),
                frequency=EQ_FREQUENCY
            )
            df = pd.DataFrame(inst.data)
            if df.empty:
                continue
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'].dt.date == (today if da == 0 else tomorrow))]
            frames.append(df.set_index('date').rename(columns={'value': col_name}))
        return pd.concat(frames).sort_index() if frames else pd.DataFrame(columns=[col_name])

    temperature = load('GR Consumption Temperature °C 15min Forecast', 'Temperature')
    solar       = load('GR Solar Photovoltaic Production MWh/h 15min Forecast', 'Solar')
    wind        = load('GR Wind Power Production MWh/h 15min Forecast', 'Wind')
    load        = load('GR Consumption MWh/h 15min Forecast', 'Consumption')

    df = pd.concat([temperature, solar, wind, load], axis=1).reset_index()
    df.rename(columns={'index': 'date'}, inplace=True)

    # 15' time parts
    df['Date'] = df['date'].dt.date
    df['Hour'] = df['date'].dt.hour.astype('int16')
    df['Minute'] = df['date'].dt.minute.astype('int16')
    df['Quarter'] = (df['Minute'] // 15).astype('int16')

    # Save quick check
    df[['Date','Hour','Minute','Quarter','Temperature','Solar','Wind','Consumption']].to_csv('temp1.csv', sep=';', index=False)
    print("[EQ] temp1.csv written.")

    return df

# ==========================================
# 3) FEATURE ENGINEERING (FAST)
# ==========================================
def _to_float_fast(s: pd.Series) -> pd.Series:
    if s.dtype == 'O':
        s = s.astype(str).str.replace(',', '.', regex=False)
    return pd.to_numeric(s, errors='coerce').astype('float32')

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if 'Date' in out.columns and not np.issubdtype(out['Date'].dtype, np.datetime64):
        out['Date'] = pd.to_datetime(out['Date'], errors='coerce')

    # sanitize numeric cols
    for c in ['Production','Temperature','Solar','Wind','Consumption','Capacity']:
        if c in out.columns:
            out[c] = _to_float_fast(out[c])

    # ensure ints small dtypes
    for c in ['Hour','Minute','Quarter']:
        if c not in out.columns:
            out[c] = 0
        out[c] = out[c].astype('int16', errors='ignore')

    # Calendar
    out['Weekday'] = out['Date'].dt.weekday.astype('int8')
    out['Is_Weekend'] = (out['Weekday'] >= 5).astype('int8')
    out['Month'] = out['Date'].dt.month.astype('int8')

    # Season
    month = out['Month'].values
    season = np.where(np.isin(month, [12,1,2]), 0,
              np.where(np.isin(month, [3,4,5]), 1,
              np.where(np.isin(month, [6,7,8]), 2, 3)))
    out['Season'] = season.astype('int8')

    # Bins
    h = out['Hour'].astype('int16')
    out['Hour_Bin'] = np.select(
        [h < 6, (h >= 6) & (h < 12), (h >= 12) & (h < 18)],
        [0, 1, 2], default=3).astype('int8')

    t = out['Temperature'].astype('float32')
    out['Temp_Bin'] = np.select([t < 10, t < 20], [0, 1], default=2).astype('int8')

    # Interactions (incl. Capacity)
    out['Solar_Temp'] = (out['Solar'] * t).astype('float32')
    out['Wind_Temp']  = (out['Wind']  * t).astype('float32')
    out['Cons_Temp']  = (out['Consumption'] * t).astype('float32')
    out['Hour_Solar'] = (h * out['Solar']).astype('float32')
    out['Weekday_Consumption'] = (out['Weekday'] * out['Consumption']).astype('float32')

    out['Hour_sin'] = np.sin(2 * np.pi * h / 24).astype('float32')
    out['Hour_cos'] = np.cos(2 * np.pi * h / 24).astype('float32')
    out['Ordinal_Date'] = out['Date'].map(pd.Timestamp.toordinal).astype('int32')

    # Ratios
    out['Load_Solar_Ratio'] = (out['Consumption'] / (out['Solar'] + 1.0)).astype('float32')
    out['Load_Wind_Ratio']  = (out['Consumption'] / (out['Wind'] + 1.0)).astype('float32')

    # Relative Solar per day
    daily_max = out.groupby(out['Date'].dt.date, sort=False)['Solar'].transform('max').astype('float32')
    out['Solar_Relative'] = (out['Solar'] / (daily_max + 1.0)).astype('float32')

    out['Season_Hour'] = (out['Season'].astype('int16') * h).astype('int16')
    out['Is_Peak_Hour'] = ((h >= 11) & (h <= 17)).astype('int8')

    # Daily cons stats
    grp = out.groupby(out['Date'].dt.date, sort=False)['Consumption']
    out['Daily_Max_Consumption'] = grp.transform('max').astype('float32')
    out['Daily_Min_Consumption'] = grp.transform('min').astype('float32')

    # Polynomials + Capacity
    if 'Capacity' not in out.columns:
        out['Capacity'] = np.nan
    out['Capacity'] = out['Capacity'].fillna(method='ffill').fillna(0).astype('float32')
    out['Temperature_sq'] = (t * t).astype('float32')
    out['Consumption_sq'] = (out['Consumption'] * out['Consumption']).astype('float32')
    out['Capacity_sq']    = (out['Capacity'] * out['Capacity']).astype('float32')
    out['Load_Capacity_Ratio']  = (out['Consumption'] / (out['Capacity'] + 1.0)).astype('float32')
    out['Solar_Capacity_Ratio'] = (out['Solar'] / (out['Capacity'] + 1.0)).astype('float32')
    out['Cap_Temp'] = (out['Capacity'] * t).astype('float32')

    # PV-focused
    out['Quarter_In_Day'] = (h * 4 + out['Quarter']).astype('int16')
    out['Is_Daylight'] = (out['Solar'] >= PV_SOLAR_DAYLIGHT_THRESHOLD).astype('int8')

    # 1st derivatives
    for c in ['Solar','Wind','Consumption','Temperature']:
        out[f'd{c}'] = out[c].diff().fillna(0).astype('float32')

    # Fast rolling (window=3)
    for c in ['Solar','Wind','Consumption','Temperature','Capacity']:
        out[f'{c}_roll3_mean'] = out[c].rolling(3, min_periods=1).mean().astype('float32')
        out[f'{c}_roll3_std']  = out[c].rolling(3, min_periods=1).std().fillna(0).astype('float32')

    # sameHour rolling mean over last 7 samples per HOUR (fast transform)
    out = out.sort_values(['Date','Hour','Minute','Quarter'])
    for c in ['Solar','Wind','Consumption','Temperature','Capacity']:
        out[f'{c}_sameHour_roll7_mean'] = (
            out.groupby('Hour', sort=False)[c]
               .transform(lambda s: s.rolling(7, min_periods=1).mean())
               .astype('float32')
        )

    # cumulative solar per day
    out['Solar_cum_day'] = out.groupby(out['Date'].dt.date, sort=False)['Solar'].cumsum().astype('float32')

    # 2nd deriv
    out['ddSolar'] = out['dSolar'].diff().fillna(0).astype('float32')

    return out

def add_lags_rollings(train_df: pd.DataFrame) -> pd.DataFrame:
    tr = train_df.copy()
    for lag in (1, 2, 3):
        tr[f'Consumption_lag{lag}'] = tr['Consumption'].shift(lag)
        tr[f'Solar_lag{lag}']       = tr['Solar'].shift(lag)
        tr[f'Wind_lag{lag}']        = tr['Wind'].shift(lag)
        tr[f'Capacity_lag{lag}']    = tr['Capacity'].shift(lag)
    tr['Consumption_roll3_mean'] = tr['Consumption'].rolling(3).mean()
    tr['Consumption_roll3_std']  = tr['Consumption'].rolling(3).std()
    return tr.fillna(0)

def detect_training_granularity(ermis_df: pd.DataFrame) -> str:
    if 'Date' not in ermis_df.columns or 'Hour' not in ermis_df.columns:
        return 'hourly'
    tmp = ermis_df.copy()
    if not np.issubdtype(tmp['Date'].dtype, np.datetime64):
        tmp['Date'] = pd.to_datetime(tmp['Date'], errors='coerce')
    per_day = tmp.groupby(tmp['Date'].dt.date, sort=False).size()
    return 'quarter' if (not per_day.empty and per_day.median() > 24) else 'hourly'

def resample_15m_to_hourly(df_15m: pd.DataFrame) -> pd.DataFrame:
    df = df_15m.copy()
    if 'date' not in df.columns:
        df['date'] = pd.to_datetime(df['Date']) + pd.to_timedelta(df['Hour'], unit='h') + pd.to_timedelta(df.get('Minute', 0), unit='m')
    df = df.set_index('date').sort_index()
    agg = df[['Temperature','Solar','Wind','Consumption']].resample('H').mean()
    out = agg.reset_index()
    out['Date'] = out['date'].dt.date
    out['Hour'] = out['date'].dt.hour.astype('int16')
    out['Minute'] = 0
    out['Quarter'] = 0
    return out[['Date','Hour','Minute','Quarter','Temperature','Solar','Wind','Consumption']]

# ---------- robust metrics ----------
def safe_mape(y_true, y_pred, eps=EPSILON):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

def smape(y_true, y_pred, eps=EPSILON):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    return np.mean(np.abs(y_pred - y_true) / denom) * 100.0

def wmape(y_true, y_pred, eps=EPSILON):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    num = np.sum(np.abs(y_true - y_pred))
    den = np.maximum(np.sum(np.abs(y_true)), eps)
    return (num / den) * 100.0

def accepts_sample_weight(est) -> bool:
    try:
        return 'sample_weight' in signature(est.fit).parameters
    except (ValueError, TypeError):
        return False

# ==================================
# 4) TRAIN / PREDICT / EVALUATE
# ==================================
def train_and_predict(ermis_path: str, ermis_sheet: str, eq_api_key: str) -> None:
    # 4.1 EQ 15'
    eq_df_15m = fetch_eq_forecasts_15m(eq_api_key)
    eq_df_15m['date'] = pd.to_datetime(eq_df_15m['date'])

    # 4.2 ERMIS
    ermis_df = pd.read_excel(ermis_path, sheet_name=ermis_sheet)
    ermis_df.rename(columns=lambda x: str(x).strip(), inplace=True)
    for c in list(ermis_df.columns):
        if c.lower().startswith('capaci'): ermis_df.rename(columns={c: 'Capacity'}, inplace=True)
        if c.lower().startswith('product'): ermis_df.rename(columns={c: 'Production'}, inplace=True)
    ermis_df['Date'] = pd.to_datetime(ermis_df['Date'], errors='coerce')
    for col in ['Production','Capacity','Temperature','Solar','Wind','Consumption']:
        if col in ermis_df.columns:
            ermis_df[col] = _to_float_fast(ermis_df[col])

    # 4.3 Granularity
    train_gran = detect_training_granularity(ermis_df)
    print(f"[INFO] ERMIS training granularity: {train_gran}")

    # 4.4 Make inference DF comparable to training
    if train_gran == 'hourly':
        temp_df = resample_15m_to_hourly(eq_df_15m)
    else:
        temp_df = eq_df_15m[['Date','Hour','Minute','Quarter','Temperature','Solar','Wind','Consumption']].copy()

    # Capacity for inference (use last known from ERMIS)
    last_cap = ermis_df['Capacity'].dropna().iloc[-1] if 'Capacity' in ermis_df.columns and ermis_df['Capacity'].notna().any() else 0.0
    temp_df['Capacity'] = np.float32(last_cap)

    # 4.5 Features
    ermis_fe = add_features(ermis_df)
    temp_fe  = add_features(temp_df)

    max_cons = float(ermis_fe['Consumption'].max() or 1.0)
    ermis_fe['Normalized_Consumption'] = (ermis_fe['Consumption'] / max_cons).astype('float32')
    temp_fe['Normalized_Consumption']  = (temp_fe['Consumption'] / max_cons).astype('float32')

    ermis_fe = add_lags_rollings(ermis_fe)

    # 4.6 Feature set
    required = [
        'Ordinal_Date','Weekday','Is_Weekend','Month','Season',
        'Hour','Hour_Bin','Minute','Quarter',
        'Temperature','Temp_Bin','Solar','Wind','Consumption','Capacity',
        'Solar_Temp','Wind_Temp','Cons_Temp','Cap_Temp',
        'Hour_Solar','Weekday_Consumption',
        'Hour_sin','Hour_cos','Normalized_Consumption',
        'Consumption_lag1','Consumption_lag2','Consumption_lag3',
        'Solar_lag1','Solar_lag2','Solar_lag3',
        'Wind_lag1','Wind_lag2','Wind_lag3',
        'Capacity_lag1','Capacity_lag2','Capacity_lag3',
        'Consumption_roll3_mean','Consumption_roll3_std',
        'Load_Solar_Ratio','Load_Wind_Ratio','Solar_Relative',
        'Season_Hour','Is_Peak_Hour',
        'Daily_Max_Consumption','Daily_Min_Consumption',
        'Temperature_sq','Consumption_sq','Capacity_sq',
        'Quarter_In_Day','Is_Daylight',
        'dSolar','dWind','dConsumption','dTemperature',
        'Solar_roll3_mean','Solar_roll3_std',
        'Wind_roll3_mean','Wind_roll3_std',
        'Temperature_roll3_mean','Temperature_roll3_std',
        'Capacity_roll3_mean','Capacity_roll3_std',
        'Solar_sameHour_roll7_mean','Wind_sameHour_roll7_mean',
        'Consumption_sameHour_roll7_mean','Temperature_sameHour_roll7_mean',
        'Capacity_sameHour_roll7_mean',
        'Solar_cum_day','ddSolar',
        'Load_Capacity_Ratio','Solar_Capacity_Ratio'
    ]
    for c in required:
        if c not in ermis_fe.columns: ermis_fe[c] = 0
        if c not in temp_fe.columns:  temp_fe[c]  = 0

    # X / y
    X_df = ermis_fe[required].astype('float32')
    y_ser = ermis_fe['Production'].astype('float32')
    sample_weights = (1.0 + (ermis_fe['Is_Daylight'].astype('float32') * (DAY_WEIGHT - 1.0))).values

    X_np = X_df.fillna(0).to_numpy(dtype='float32')
    y_np = y_ser.values.astype('float32')
    is_day = ermis_fe['Is_Daylight'].astype(bool).values

    X_train_np, X_test_np, y_train, y_test, w_train, w_test, day_tr, day_te = train_test_split(
        X_np, y_np, sample_weights, is_day, test_size=0.2, random_state=RANDOM_STATE
    )

    # 4.8 GridSearch (cv=3, parallel)
    param_grid_rf = {
        'n_estimators': [80, 120],
        'max_depth': [5, 7],
        'min_samples_split': [10],
        'min_samples_leaf': [5],
        'bootstrap': [True],
        'n_jobs': [-1]
    }
    param_grid_gb = {
        'n_estimators': [60, 100],
        'max_depth': [3, 5],
        'min_samples_split': [10],
        'min_samples_leaf': [5],
        'subsample': [0.8, 1.0]
    }
    param_grid_xgb = {
        'n_estimators': [80, 120],
        'max_depth': [4, 5],
        'min_child_weight': [5],
        'alpha': [0.1, 0.3],
        'reg_lambda': [0.1, 0.3],
        'learning_rate': [0.05, 0.1],
        'tree_method': ['hist'],
        'n_jobs': [-1],
        'random_state': [RANDOM_STATE]
    }

    models = {}
    for name, model, grid in zip(
        ['RandomForest', 'GradientBoosting', 'XGBoost'],
        [RandomForestRegressor(random_state=RANDOM_STATE),
         GradientBoostingRegressor(random_state=RANDOM_STATE),
         XGBRegressor(random_state=RANDOM_STATE)],
        [param_grid_rf, param_grid_gb, param_grid_xgb]
    ):
        print(f"[GRID] {name}...")
        gs = GridSearchCV(model, grid, cv=3, scoring='r2', n_jobs=-1, verbose=0)
        fit_kwargs = {}
        if accepts_sample_weight(model):
            fit_kwargs['sample_weight'] = w_train
        gs.fit(X_train_np, y_train, **fit_kwargs)
        print(f"[GRID] {name} best params: {gs.best_params_}")
        models[name] = gs.best_estimator_

    # 4.9 Evaluation
    def eval_block(y_true, y_pred, mask=None):
        if mask is None:
            mask = np.ones_like(y_true, dtype=bool)
        yt, yp = y_true[mask], y_pred[mask]
        return {
            'RMSE': float(np.sqrt(mean_squared_error(yt, yp))),
            'MAE': float(mean_absolute_error(yt, yp)),
            'MedAE': float(median_absolute_error(yt, yp)),
            'MAPE%': float(safe_mape(yt, yp)),
            'SMAPE%': float(smape(yt, yp)),
            'WMAPE%': float(wmape(yt, yp)),
            'R2': float(r2_score(yt, yp)) if len(np.unique(yt)) > 1 else np.nan
        }

    results = {}
    day_mask = day_te.astype(bool)
    night_mask = ~day_mask

    for name, model in models.items():
        fit_kwargs = {}
        if accepts_sample_weight(model):
            fit_kwargs['sample_weight'] = w_train
        model.fit(X_train_np, y_train, **fit_kwargs)
        y_pred = np.maximum(model.predict(X_test_np), 0)

        full = eval_block(y_test, y_pred)
        day  = eval_block(y_test, y_pred, day_mask)
        night= eval_block(y_test, y_pred, night_mask)

        print(f"[METRICS] {name} (ALL):   RMSE={full['RMSE']:.3f}  MAE={full['MAE']:.3f}  R²={full['R2']:.3f}  "
              f"MAPE={full['MAPE%']:.2f}%  SMAPE={full['SMAPE%']:.2f}%  WMAPE={full['WMAPE%']:.2f}%")
        print(f"[METRICS] {name} (DAY):   RMSE={day['RMSE']:.3f}  MAE={day['MAE']:.3f}  R²={day['R2']:.3f}  "
              f"MAPE={day['MAPE%']:.2f}%  SMAPE={day['SMAPE%']:.2f}%  WMAPE={day['WMAPE%']:.2f}%")
        print(f"[METRICS] {name} (NIGHT): RMSE={night['RMSE']:.3f} MAE={night['MAE']:.3f} R²={night['R2']:.3f} "
              f"MAPE={night['MAPE%']:.2f}% SMAPE={night['SMAPE%']:.2f}% WMAPE={night['WMAPE%']:.2f}%")

        results[name] = {'model': model, 'y_pred': y_pred, 'full': full, 'day': day, 'night': night}

    # 4.10 Predict on inference set
    def predict_df(df: pd.DataFrame, model, features: list[str]) -> np.ndarray:
        X_pred = df[features].fillna(0).astype('float32').to_numpy()
        y_hat = model.predict(X_pred)
        return np.maximum(y_hat, 0)

    final_features = X_df.columns.tolist()
    for name, res in results.items():
        temp_fe[f'Predicted_Production_{name}'] = predict_df(temp_fe, res['model'], final_features)

    # 4.11 Export
    out_cols = ['Date','Hour','Minute','Quarter','Temperature','Solar','Wind','Consumption','Capacity'] + \
               [f'Predicted_Production_{n}' for n in results.keys()]
    out_df = temp_fe[out_cols].copy()
    out_path = 'predictions_temperature_hour_wind_consumption_capacity_fast.xlsx'
    out_df.to_excel(out_path, index=False)
    print(f"[OK] Saved: {out_path}")

    # 4.12 Charts (all kept; use smaller samples for speed where needed)
    # Histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(y_np, bins=30, kde=True, color='blue')
    plt.title('Distribution of Production'); plt.xlabel('Production'); plt.ylabel('Frequency'); plt.grid(True)

    # Correlation
    plt.figure(figsize=(10, 8))
    corr_matrix = ermis_fe.select_dtypes(include=[np.number]).corr(numeric_only=True)
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Features')

    # Learning curves (fewer points; on sample to speed up)
    lc_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_np), size=min(8000, len(X_np)), replace=False)
    X_lc, y_lc = X_np[lc_idx], y_np[lc_idx]
    for name, res in results.items():
        plt.figure(figsize=(8, 6))
        train_sizes, train_scores, test_scores = learning_curve(
            res['model'], X_lc, y_lc, cv=3, scoring='r2', train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=-1
        )
        plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training', color='blue')
        plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Validation', color='green')
        plt.title(f'Learning Curve – {name}')
        plt.xlabel('Training examples'); plt.ylabel('R²'); plt.legend(); plt.grid(True)

    # Residuals (sampled)
    for name, res in results.items():
        resid = y_test - res['y_pred']
        if len(resid) > 3000:
            sel = np.random.RandomState(RANDOM_STATE).choice(len(resid), size=3000, replace=False)
            resid = resid[sel]
        plt.figure(figsize=(8, 6))
        plt.scatter(range(len(resid)), resid, color='green', alpha=0.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.title(f'Residuals – {name}')
        plt.xlabel('Test Sample Index'); plt.ylabel('Actual - Predicted'); plt.grid(True)

    # Summary metrics table (print)
    metrics_rows = []
    for name, res in results.items():
        metrics_rows += [
            {'Model': name, 'Split':'ALL', **res['full']},
            {'Model': name, 'Split':'DAY', **res['day']},
            {'Model': name, 'Split':'NIGHT', **res['night']},
        ]
    metrics_df = pd.DataFrame(metrics_rows)
    print("\n[SUMMARY METRICS]\n", metrics_df)
    print(out_df.tail(3))

# --------------------------
# main
# --------------------------
if __name__ == "__main__":
    train_and_predict(ERMIS_XLSX, ERMIS_SHEET, EQ_API_KEY)




