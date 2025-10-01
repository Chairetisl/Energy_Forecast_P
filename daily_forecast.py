#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

# ========= (2) Thread caps ΠΡΙΝ από numpy/pandas =========
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

from datetime import datetime, time, timedelta
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

# --- EnergyQuantified ---
from energyquantified import EnergyQuantified
from energyquantified.time import Frequency

# --- ML ---
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from inspect import signature

# (1) HalvingRandomSearchCV (πολύ πιο γρήγορο από GridSearch)
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV


# =========================
# 1) ΡΥΘΜΙΣΕΙΣ / CONFIG
# =========================
EQ_API_KEY = "9b1b06b8-0074f355-02550874-2703fde3"  # <-- βάλε το δικό σου αν χρειάζεται
ERMIS_XLSX = "ERMIS_II - Real1Q.xlsx"               # sheet: Sheet1 (πρέπει να έχει 'Production')
ERMIS_SHEET = "Sheet1"

# Προεπιλογή: Φέρνουμε 15' από EQ
EQ_FREQUENCY = Frequency.PT15M  # PT15M για 15λεπτα, PT1H για ώρα

# Zero-inflated PV helpers
PV_SOLAR_DAYLIGHT_THRESHOLD = 5.0   # MWh/h για να θεωρήσω “μέρα”
EPSILON = 1e-6
DAY_WEIGHT = 3.0                    # ζύγισμα ημερών > νύχτες

# (1) Ρυθμίσεις για γρήγορο αλλά αποδοτικό search στο CI
CV_FOLDS = 3         # από 5 -> 3 (σημαντική μείωση χρόνου)
N_JOBS = 2           # 2 vCPU στους GitHub runners


# =========================
# 2) ΛΗΨΗ ΠΡΟΒΛΕΨΕΩΝ EQ
# =========================
def fetch_eq_forecasts_15m(api_key: str) -> pd.DataFrame:
    """
    Τραβάει 15λεπτα από EQ για Temperature / Solar / Wind / Consumption, για ΣΗΜΕΡΑ & ΑΥΡΙΟ,
    αποφεύγοντας επικαλύψεις (days_ahead=0 → μόνο σήμερα, 1 → μόνο αύριο).
    """
    eq = EnergyQuantified(api_key=api_key)

    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)
    day_after_tomorrow = today + timedelta(days=2)
    begin = datetime.combine(today, time(0, 0))
    end = datetime.combine(day_after_tomorrow, time(0, 0))

    def load(series_name: str, col_name: str) -> pd.DataFrame:
        parts = []
        for da in [0, 1]:
            inst = eq.instances.relative(
                series_name,
                begin=begin,
                end=end,
                tag='ec',
                days_ahead=da,
                time_of_day=time(0, 0),
                frequency=EQ_FREQUENCY
            )
            df = pd.DataFrame(inst.data)
            if df.empty:
                continue
            df['date'] = pd.to_datetime(df['date'])
            if da == 0:
                df = df[df['date'].dt.date == today]
            else:
                df = df[df['date'].dt.date == tomorrow]
            df = df.set_index('date').rename(columns={'value': col_name})
            parts.append(df)
        if not parts:
            return pd.DataFrame(columns=[col_name])
        return pd.concat(parts).sort_index()

    temperature = load('GR Consumption Temperature °C 15min Forecast', 'Temperature')
    solar       = load('GR Solar Photovoltaic Production MWh/h 15min Forecast', 'Solar')
    wind        = load('GR Wind Power Production MWh/h 15min Forecast', 'Wind')
    load        = load('GR Consumption MWh/h 15min Forecast', 'Consumption')

    concat = pd.concat([temperature, solar, wind, load], axis=1).reset_index()
    concat.rename(columns={'index': 'date'}, inplace=True)

    # Χρονικές στήλες για 15'
    concat['Date'] = concat['date'].dt.date
    concat['Hour'] = concat['date'].dt.hour
    concat['Minute'] = concat['date'].dt.minute
    concat['Quarter'] = (concat['Minute'] // 15).astype(int)

    # Αποθήκευση για έλεγχο
    concat[['Date','Hour','Minute','Quarter','Temperature','Solar','Wind','Consumption']].to_csv(
        'temp1.csv', sep=';', index=False
    )
    print("[EQ] temp1.csv γράφτηκε (15λεπτα).")

    return concat


# ==========================================
# 3) FEATURE ENGINEERING & HELPERS
# ==========================================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Μετατροπές τύπων
    if 'Date' in out.columns and not np.issubdtype(out['Date'].dtype, np.datetime64):
        out['Date'] = pd.to_datetime(out['Date'], errors='coerce')

    out['Weekday'] = out['Date'].dt.weekday
    out['Is_Weekend'] = (out['Weekday'] >= 5).astype(int)
    out['Month'] = out['Date'].dt.month

    def season(month: int) -> int:
        if month in [12, 1, 2]: return 0
        if month in [3, 4, 5]:  return 1
        if month in [6, 7, 8]:  return 2
        return 3
    out['Season'] = out['Month'].apply(season)

    def hour_bin(h: int) -> int:
        if 0 <= h < 6:   return 0
        if 6 <= h < 12:  return 1
        if 12 <= h < 18: return 2
        return 3
    out['Hour_Bin'] = out['Hour'].astype(int).apply(hour_bin)

    def temp_bin(t: float) -> int:
        if t < 10: return 0
        if t < 20: return 1
        return 2
    out['Temp_Bin'] = out['Temperature'].astype(float).apply(temp_bin)

    out['Solar_Temp'] = out['Solar'] * out['Temperature']
    out['Wind_Temp']  = out['Wind']  * out['Temperature']
    out['Cons_Temp']  = out['Consumption'] * out['Temperature']
    out['Hour_Solar'] = out['Hour'] * out['Solar']
    out['Weekday_Consumption'] = out['Weekday'] * out['Consumption']

    out['Hour_sin'] = np.sin(2 * np.pi * out['Hour'] / 24)
    out['Hour_cos'] = np.cos(2 * np.pi * out['Hour'] / 24)

    out['Ordinal_Date'] = out['Date'].map(pd.Timestamp.toordinal)

    out['Load_Solar_Ratio'] = out['Consumption'] / (out['Solar'] + 1)
    out['Load_Wind_Ratio']  = out['Consumption'] / (out['Wind'] + 1)

    out['Solar_Relative'] = out['Solar'] / (out.groupby(out['Date'].dt.date)['Solar'].transform('max') + 1)
    out['Season_Hour'] = out['Season'] * out['Hour']
    out['Is_Peak_Hour'] = out['Hour'].apply(lambda x: 1 if 11 <= x <= 17 else 0)

    out['Daily_Max_Consumption'] = out.groupby(out['Date'].dt.date)['Consumption'].transform('max')
    out['Daily_Min_Consumption'] = out.groupby(out['Date'].dt.date)['Consumption'].transform('min')

    out['Temperature_sq'] = out['Temperature'] ** 2
    out['Consumption_sq'] = out['Consumption'] ** 2

    if 'Minute' not in out.columns:
        out['Minute'] = 0
    if 'Quarter' not in out.columns:
        out['Quarter'] = 0

    out['Quarter_In_Day'] = (out['Hour'] * 4 + out['Quarter']).astype(int)
    out['Is_Daylight'] = (out['Solar'] >= PV_SOLAR_DAYLIGHT_THRESHOLD).astype(int)

    out['dSolar']        = out['Solar'].diff().fillna(0)
    out['dWind']         = out['Wind'].diff().fillna(0)
    out['dConsumption']  = out['Consumption'].diff().fillna(0)
    out['dTemperature']  = out['Temperature'].diff().fillna(0)

    for c in ['Solar','Wind','Consumption','Temperature']:
        out[f'{c}_roll3_mean'] = out[c].rolling(window=3, min_periods=1).mean()
        out[f'{c}_roll3_std']  = out[c].rolling(window=3, min_periods=1).std().fillna(0)

    out = out.sort_values(['Date','Hour','Minute','Quarter'])
    for c in ['Solar','Wind','Consumption','Temperature']:
        out[f'{c}_sameHour_roll7_mean'] = (
            out.groupby(out['Hour'])[c]
               .apply(lambda s: s.rolling(window=7, min_periods=1).mean())
               .reset_index(level=0, drop=True)
        )

    out['Solar_cum_day'] = out.groupby(out['Date'].dt.date)['Solar'].cumsum()
    out['ddSolar'] = out['dSolar'].diff().fillna(0)

    return out


def add_lags_rollings(train_df: pd.DataFrame) -> pd.DataFrame:
    tr = train_df.copy()
    for lag in [1, 2, 3]:
        tr[f'Consumption_lag{lag}'] = tr['Consumption'].shift(lag)
        tr[f'Solar_lag{lag}']       = tr['Solar'].shift(lag)
        tr[f'Wind_lag{lag}']        = tr['Wind'].shift(lag)
    tr['Consumption_roll3_mean'] = tr['Consumption'].rolling(window=3).mean()
    tr['Consumption_roll3_std']  = tr['Consumption'].rolling(window=3).std()
    tr.fillna(0, inplace=True)
    return tr


def detect_training_granularity(ermis_df: pd.DataFrame) -> str:
    if 'Date' not in ermis_df.columns or 'Hour' not in ermis_df.columns:
        return 'hourly'
    tmp = ermis_df.copy()
    if not np.issubdtype(tmp['Date'].dtype, np.datetime64):
        tmp['Date'] = pd.to_datetime(tmp['Date'], errors='coerce')
    tmp['_dateonly'] = tmp['Date'].dt.date
    per_day = tmp.groupby('_dateonly').size()
    if per_day.empty:
        return 'hourly'
    median_per_day = per_day.median()
    return 'quarter' if median_per_day > 24 else 'hourly'


def resample_15m_to_hourly(df_15m: pd.DataFrame) -> pd.DataFrame:
    df = df_15m.copy()
    if 'date' not in df.columns:
        df['date'] = (pd.to_datetime(df['Date'])
                      + pd.to_timedelta(df['Hour'], unit='h')
                      + pd.to_timedelta(df.get('Minute', 0), unit='m'))
    df = df.set_index('date').sort_index()

    agg = df[['Temperature','Solar','Wind','Consumption']].resample('H').mean()
    out = agg.reset_index()
    out['Date'] = out['date'].dt.date
    out['Hour'] = out['date'].dt.hour
    out['Minute'] = 0
    out['Quarter'] = 0
    return out[['Date','Hour','Minute','Quarter','Temperature','Solar','Wind','Consumption']]


def accepts_sample_weight(est) -> bool:
    try:
        return 'sample_weight' in signature(est.fit).parameters
    except (ValueError, TypeError):
        return False


# ==================================
# 4) TRAIN / PREDICT (χωρίς plots/metrics)
# ==================================
def train_and_predict(ermis_path: str, ermis_sheet: str, eq_api_key: str) -> None:
    # EQ 15'
    eq_df_15m = fetch_eq_forecasts_15m(eq_api_key)
    eq_df_15m['date'] = pd.to_datetime(eq_df_15m['date'])

    # ERMIS training
    ermis_df = pd.read_excel(ermis_path, sheet_name=ermis_sheet)
    ermis_df.rename(columns=lambda x: str(x).strip(), inplace=True)
    ermis_df['Date'] = pd.to_datetime(ermis_df['Date'], format='%Y-%m-%d', errors='coerce')

    # granular
    train_gran = detect_training_granularity(ermis_df)
    print(f"[INFO] ERMIS training granularity: {train_gran}")

    # inference set aligned to ERMIS granularity
    if train_gran == 'hourly':
        temp_df = resample_15m_to_hourly(eq_df_15m)
    else:
        temp_df = eq_df_15m[['Date','Hour','Minute','Quarter','Temperature','Solar','Wind','Consumption']].copy()

    # features
    ermis_fe = add_features(ermis_df)
    temp_fe  = add_features(temp_df)

    # normalization
    max_cons = float(ermis_fe['Consumption'].max() or 1.0)
    ermis_fe['Normalized_Consumption'] = ermis_fe['Consumption'] / max_cons
    temp_fe['Normalized_Consumption']  = temp_fe['Consumption']  / max_cons

    # lags/rollings for training only
    ermis_fe = add_lags_rollings(ermis_fe)

    required_columns = [
        'Ordinal_Date','Weekday','Is_Weekend','Month','Season',
        'Hour','Hour_Bin','Minute','Quarter',
        'Temperature','Temp_Bin','Solar','Wind','Consumption',
        'Solar_Temp','Wind_Temp','Cons_Temp',
        'Hour_Solar','Weekday_Consumption',
        'Hour_sin','Hour_cos','Normalized_Consumption',
        'Consumption_lag1','Consumption_lag2','Consumption_lag3',
        'Solar_lag1','Solar_lag2','Solar_lag3',
        'Wind_lag1','Wind_lag2','Wind_lag3',
        'Consumption_roll3_mean','Consumption_roll3_std',
        'Load_Solar_Ratio','Load_Wind_Ratio',
        'Solar_Relative','Season_Hour','Is_Peak_Hour',
        'Daily_Max_Consumption','Daily_Min_Consumption',
        'Temperature_sq','Consumption_sq',
        'Quarter_In_Day','Is_Daylight',
        'dSolar','dWind','dConsumption','dTemperature',
        'Solar_roll3_mean','Solar_roll3_std',
        'Wind_roll3_mean','Wind_roll3_std',
        'Temperature_roll3_mean','Temperature_roll3_std',
        'Solar_sameHour_roll7_mean','Wind_sameHour_roll7_mean',
        'Consumption_sameHour_roll7_mean','Temperature_sameHour_roll7_mean',
        'Solar_cum_day','ddSolar'
    ]
    for col in required_columns:
        if col not in ermis_fe.columns: ermis_fe[col] = 0
        if col not in temp_fe.columns:  temp_fe[col]  = 0

    # X/y
    X_df = ermis_fe[required_columns]
    y_ser = ermis_fe['Production'].astype(float)

    # weights
    sample_weights = 1.0 + (ermis_fe['Is_Daylight'].astype(float) * (DAY_WEIGHT - 1.0))
    sample_weights = np.asarray(sample_weights, dtype=float).ravel()

    # to numpy
    final_features = X_df.columns.tolist()
    X_np = X_df.fillna(0).to_numpy()
    y_np = np.asarray(y_ser, dtype=float).ravel()
    is_day = np.asarray(ermis_fe['Is_Daylight'].astype(bool).values)

    X_train_np, X_test_np, y_train, y_test, w_train, w_test, day_tr, day_te = train_test_split(
        X_np, y_np, sample_weights, is_day, test_size=0.2, random_state=42
    )

    # --- HalvingRandomSearchCV params ---
    param_dist_rf = {
        'n_estimators': [50, 80, 120, 160],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf': [3, 5, 10],
        'bootstrap': [True]
    }
    param_dist_gb = {
        'n_estimators': [40, 80, 120, 160],
        'max_depth': [3, 4, 5],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf': [3, 5, 10],
        'subsample': [0.7, 0.85, 1.0]
    }
    param_dist_xgb = {
        'n_estimators': [80, 120, 160, 220],
        'max_depth': [3, 4, 5],
        'min_child_weight': [3, 5, 7],
        'alpha': [0.0, 0.1, 0.3],
        'reg_lambda': [0.1, 0.3, 1.0],
        'learning_rate': [0.05, 0.1],
        'tree_method': ['hist'],     # γρήγορο
        'n_jobs': [N_JOBS]
    }

    def make_search(est, dist):
        return HalvingRandomSearchCV(
            estimator=est,
            param_distributions=dist,
            factor=3,
            resource='n_samples',
            min_resources='smallest',   # FIX (όχι 'exhaust')
            cv=CV_FOLDS,
            n_jobs=N_JOBS,
            scoring='r2',
            random_state=42
        )

    searches = {
        'RandomForest':     make_search(RandomForestRegressor(random_state=42),        param_dist_rf),
        'GradientBoosting': make_search(GradientBoostingRegressor(random_state=42),    param_dist_gb),
        'XGBoost':          make_search(XGBRegressor(random_state=42, verbosity=0),    param_dist_xgb),
    }

    # fit + refit on full data
    models = {}
    for name, search in searches.items():
        if accepts_sample_weight(search.estimator):
            search.fit(X_train_np, y_train, sample_weight=w_train)
        else:
            search.fit(X_train_np, y_train)
        best_est = search.best_estimator_
        if accepts_sample_weight(best_est):
            best_est.fit(X_np, y_np, sample_weight=sample_weights)
        else:
            best_est.fit(X_np, y_np)
        models[name] = best_est

    # inference
    def predict_df(df: pd.DataFrame, model, features: list[str]) -> np.ndarray:
        df2 = df.copy()
        for col in features:
            if col not in df2.columns:
                df2[col] = 0
        X_pred = df2[features].fillna(0).to_numpy()
        y_hat = model.predict(X_pred)
        return np.maximum(y_hat, 0)

    for name, model in models.items():
        temp_fe[f'Predicted_Production_{name}'] = predict_df(temp_fe, model, final_features)

    # export
    out_cols = ['Date','Hour','Minute','Quarter','Temperature','Solar','Wind','Consumption'] + \
               [f'Predicted_Production_{n}' for n in models.keys()]
    out_df = temp_fe[out_cols].copy()
    out_path = 'predictions_temperature_hour_wind_consumption.xlsx'
    out_df.to_excel(out_path, index=False)
    print(f"[OK] Αποθηκεύτηκε: {out_path}")
    print(out_df.tail(3))


# --------------------------
# main
# --------------------------
if __name__ == "__main__":
    train_and_predict(ERMIS_XLSX, ERMIS_SHEET, EQ_API_KEY)

    # --- Προαιρετικά: email αποστολή αρχείου (Office365) ---
    """
    import smtplib
    from email.message import EmailMessage

    SENDER = "l.chairetis@depa.gr"
    APP_PASSWORD = "********"  # Χρησιμοποίησε App Password αν έχεις MFA
    RECEIVER = "trading.power@depa.gr"

    msg = EmailMessage()
    msg["Subject"] = "Predictions File"
    msg["From"] = SENDER
    msg["To"] = RECEIVER
    msg.set_content("Επισυνάπτεται το αρχείο με τις προβλέψεις παραγωγής.")

    with open("predictions_temperature_hour_wind_consumption.xlsx", "rb") as f:
        data = f.read()
        name = "predictions_temperature_hour_wind_consumption.xlsx"
    msg.add_attachment(
        data,
        maintype="application",
        subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=name
    )

    with smtplib.SMTP("smtp.office365.com", 587) as smtp:
        smtp.starttls()
        smtp.login(SENDER, APP_PASSWORD)
        smtp.send_message(msg)

    print("Email sent successfully.")
    """



