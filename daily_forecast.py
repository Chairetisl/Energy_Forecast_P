#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
EQ_API_KEY = "9b1b06b8-0074f355-02550874-2703fde3"  # <-- βάλε το δικό σου αν χρειάζεται
ERMIS_XLSX = "ERMIS_II - Real1Q.xlsx"               # sheet: Sheet1 (πρέπει να έχει 'Production')
ERMIS_SHEET = "Sheet1"

# Προεπιλογή: Φέρνουμε 15' από EQ
EQ_FREQUENCY = Frequency.PT15M  # PT15M για 15λεπτα, PT1H για ώρα

# Zero-inflated PV helpers
PV_SOLAR_DAYLIGHT_THRESHOLD = 5.0   # MWh/h για να θεωρήσω “μέρα”
EPSILON = 1e-6
DAY_WEIGHT = 3.0                    # ζύγισμα ημερών > νύχτες


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

    # Calendar
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

    # Interactions
    out['Solar_Temp'] = out['Solar'] * out['Temperature']
    out['Wind_Temp']  = out['Wind']  * out['Temperature']
    out['Cons_Temp']  = out['Consumption'] * out['Temperature']
    out['Hour_Solar'] = out['Hour'] * out['Solar']
    out['Weekday_Consumption'] = out['Weekday'] * out['Consumption']

    # Sin/Cos Hour
    out['Hour_sin'] = np.sin(2 * np.pi * out['Hour'] / 24)
    out['Hour_cos'] = np.cos(2 * np.pi * out['Hour'] / 24)

    # Ordinal
    out['Ordinal_Date'] = out['Date'].map(pd.Timestamp.toordinal)

    # Ratios
    out['Load_Solar_Ratio'] = out['Consumption'] / (out['Solar'] + 1)
    out['Load_Wind_Ratio']  = out['Consumption'] / (out['Wind'] + 1)

    # Relative Solar (ημερήσια)
    out['Solar_Relative'] = out['Solar'] / (out.groupby(out['Date'].dt.date)['Solar'].transform('max') + 1)

    # Season * Hour
    out['Season_Hour'] = out['Season'] * out['Hour']

    # Peak flag
    out['Is_Peak_Hour'] = out['Hour'].apply(lambda x: 1 if 11 <= x <= 17 else 0)

    # Ημερήσια max/min κατανάλωση
    out['Daily_Max_Consumption'] = out.groupby(out['Date'].dt.date)['Consumption'].transform('max')
    out['Daily_Min_Consumption'] = out.groupby(out['Date'].dt.date)['Consumption'].transform('min')

    # Πολυώνυμα
    out['Temperature_sq'] = out['Temperature'] ** 2
    out['Consumption_sq'] = out['Consumption'] ** 2

    # Αν δεν υπάρχουν Minute/Quarter, κράτα τα
    if 'Minute' not in out.columns:
        out['Minute'] = 0
    if 'Quarter' not in out.columns:
        out['Quarter'] = 0

    # ---- PV-focused features ----
    out['Quarter_In_Day'] = (out['Hour'] * 4 + out['Quarter']).astype(int)
    out['Is_Daylight'] = (out['Solar'] >= PV_SOLAR_DAYLIGHT_THRESHOLD).astype(int)

    # 1ο παράγωγο
    out['dSolar']        = out['Solar'].diff().fillna(0)
    out['dWind']         = out['Wind'].diff().fillna(0)
    out['dConsumption']  = out['Consumption'].diff().fillna(0)
    out['dTemperature']  = out['Temperature'].diff().fillna(0)

    # Rolling στα 15'
    for c in ['Solar','Wind','Consumption','Temperature']:
        out[f'{c}_roll3_mean'] = out[c].rolling(window=3, min_periods=1).mean()
        out[f'{c}_roll3_std']  = out[c].rolling(window=3, min_periods=1).std().fillna(0)

    # “ίδια ώρα της ημέρας” (pattern ανά ώρα)
    out = out.sort_values(['Date','Hour','Minute','Quarter'])
    for c in ['Solar','Wind','Consumption','Temperature']:
        out[f'{c}_sameHour_roll7_mean'] = (
            out.groupby(out['Hour'])[c]
               .apply(lambda s: s.rolling(window=7, min_periods=1).mean())
               .reset_index(level=0, drop=True)
        )

    # Σωρευτικό Solar ανά ημέρα
    out['Solar_cum_day'] = out.groupby(out['Date'].dt.date)['Solar'].cumsum()

    # 2ο παράγωγο
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
    """
    Ελέγχει αν το ERMIS είναι ωριαίο ή 15λεπτο βάσει count ανά ημέρα.
    Αν ο διάμεσος αριθμός δειγμάτων/ημέρα > 24 ⇒ 15λεπτα, αλλιώς ώρα.
    """
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
    """
    Παίρνει 15λεπτα με στήλη 'date' και κάνει ωριαίο μέσο όρο για
    Temperature / Solar / Wind / Consumption.
    """
    df = df_15m.copy()
    if 'date' not in df.columns:
        df['date'] = pd.to_datetime(df['Date']) + pd.to_timedelta(df['Hour'], unit='h') + pd.to_timedelta(df.get('Minute', 0), unit='m')
    df = df.set_index('date').sort_index()

    agg = df[['Temperature','Solar','Wind','Consumption']].resample('H').mean()

    out = agg.reset_index()
    out['Date'] = out['date'].dt.date
    out['Hour'] = out['date'].dt.hour
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
    """Ελέγχει αν ο estimator.fit δέχεται sample_weight."""
    try:
        return 'sample_weight' in signature(est.fit).parameters
    except (ValueError, TypeError):
        return False


# ==================================
# 4) TRAIN / PREDICT / EVALUATE
# ==================================
def train_and_predict(ermis_path: str, ermis_sheet: str, eq_api_key: str) -> None:
    # 4.1 Φόρτωση EQ (15')
    eq_df_15m = fetch_eq_forecasts_15m(eq_api_key)
    eq_df_15m['date'] = pd.to_datetime(eq_df_15m['date'])

    # 4.2 Φόρτωση ERMIS (training)
    ermis_df = pd.read_excel(ermis_path, sheet_name=ermis_sheet)
    ermis_df.rename(columns=lambda x: str(x).strip(), inplace=True)
    ermis_df['Date'] = pd.to_datetime(ermis_df['Date'], format='%Y-%m-%d', errors='coerce')

    # 4.3 Ανίχνευση συχνότητας ERMIS
    train_gran = detect_training_granularity(ermis_df)
    print(f"[INFO] ERMIS training granularity: {train_gran}")

    # 4.4 Προετοιμασία inference set (temp_df) ώστε να ταιριάζει με ERMIS
    if train_gran == 'hourly':
        temp_df = resample_15m_to_hourly(eq_df_15m)
    else:
        temp_df = eq_df_15m[['Date','Hour','Minute','Quarter','Temperature','Solar','Wind','Consumption']].copy()

    # 4.5 Feature engineering
    ermis_fe = add_features(ermis_df)
    temp_fe  = add_features(temp_df)

    # Normalized consumption βάσει training
    max_cons = float(ermis_fe['Consumption'].max() or 1.0)
    ermis_fe['Normalized_Consumption'] = ermis_fe['Consumption'] / max_cons
    temp_fe['Normalized_Consumption']  = temp_fe['Consumption']  / max_cons

    # Lags/Rollings μόνο στο training
    ermis_fe = add_lags_rollings(ermis_fe)

    # 4.6 Feature set
    required_columns = [
        'Ordinal_Date', 'Weekday', 'Is_Weekend', 'Month', 'Season',
        'Hour', 'Hour_Bin', 'Minute', 'Quarter',
        'Temperature', 'Temp_Bin', 'Solar', 'Wind', 'Consumption',
        'Solar_Temp', 'Wind_Temp', 'Cons_Temp',
        'Hour_Solar', 'Weekday_Consumption',
        'Hour_sin', 'Hour_cos',
        'Normalized_Consumption',
        'Consumption_lag1', 'Consumption_lag2', 'Consumption_lag3',
        'Solar_lag1', 'Solar_lag2', 'Solar_lag3',
        'Wind_lag1', 'Wind_lag2', 'Wind_lag3',
        'Consumption_roll3_mean', 'Consumption_roll3_std',
        'Load_Solar_Ratio', 'Load_Wind_Ratio',
        'Solar_Relative',
        'Season_Hour',
        'Is_Peak_Hour',
        'Daily_Max_Consumption', 'Daily_Min_Consumption',
        'Temperature_sq', 'Consumption_sq',
        # PV-focused:
        'Quarter_In_Day', 'Is_Daylight',
        'dSolar','dWind','dConsumption','dTemperature',
        'Solar_roll3_mean','Solar_roll3_std',
        'Wind_roll3_mean','Wind_roll3_std',
        'Temperature_roll3_mean','Temperature_roll3_std',
        'Solar_sameHour_roll7_mean','Wind_sameHour_roll7_mean',
        'Consumption_sameHour_roll7_mean','Temperature_sameHour_roll7_mean',
        'Solar_cum_day','ddSolar'
    ]

    # Εξασφάλιση στηλών
    for col in required_columns:
        if col not in ermis_fe.columns:
            ermis_fe[col] = 0
        if col not in temp_fe.columns:
            temp_fe[col] = 0

    # Χ-ψ
    X_df = ermis_fe[required_columns]
    y_ser = ermis_fe['Production'].astype(float)

    # sample weights (ημέρα βαρύτερη)
    sample_weights = 1.0 + (ermis_fe['Is_Daylight'].astype(float) * (DAY_WEIGHT - 1.0))
    sample_weights = np.asarray(sample_weights, dtype=float).ravel()

    # --- Μετατροπή σε numpy για scikit-learn ---
    final_features = X_df.columns.tolist()
    X_np = X_df.fillna(0).to_numpy()
    y_np = np.asarray(y_ser, dtype=float).ravel()

    # Για συνεπή DAY/NIGHT mask στο split
    is_day = np.asarray(ermis_fe['Is_Daylight'].astype(bool).values)

    X_train_np, X_test_np, y_train, y_test, w_train, w_test, day_tr, day_te = train_test_split(
        X_np, y_np, sample_weights, is_day, test_size=0.2, random_state=42
    )

    # Κρατάω DataFrame μόνο για ονόματα/διευκόλυνση (όχι για fit)
    X_train = pd.DataFrame(X_train_np, columns=final_features)
    X_test  = pd.DataFrame(X_test_np,  columns=final_features)

    # 4.8 GridSearch
    param_grid_rf = {
        'n_estimators': [50, 100],
        'max_depth': [3, 7],
        'min_samples_split': [10, 15],
        'min_samples_leaf': [5, 10],
        'bootstrap': [True]
    }
    param_grid_gb = {
        'n_estimators': [30, 80],
        'max_depth': [3, 5],
        'min_samples_split': [10, 15],
        'min_samples_leaf': [5, 10],
        'subsample': [0.7, 1.0]
    }
    param_grid_xgb = {
        'n_estimators': [30, 80],
        'max_depth': [3, 5],
        'min_child_weight': [5, 7],
        'alpha': [0.1, 0.3],
        'reg_lambda': [0.1, 0.3],
        'learning_rate': [0.05, 0.1]
    }

    models = {}
    for name, model, grid in zip(
        ['RandomForest', 'GradientBoosting', 'XGBoost'],
        [RandomForestRegressor(random_state=42),
         GradientBoostingRegressor(random_state=42),
         XGBRegressor(random_state=42)],
        [param_grid_rf, param_grid_gb, param_grid_xgb]
    ):
        print(f"[GRID] {name}...")
        gs = GridSearchCV(model, grid, cv=5, scoring='r2', n_jobs=-1)
        fit_kwargs = {}
        if accepts_sample_weight(model):
            fit_kwargs['sample_weight'] = w_train  # 1D ndarray
        gs.fit(X_train_np, y_train, **fit_kwargs)  # numpy arrays only
        print(f"[GRID] {name} best params: {gs.best_params_}")
        models[name] = gs.best_estimator_

    # 4.9 Αξιολόγηση
    results = {}

    def eval_block(y_true, y_pred, mask=None):
        if mask is None:
            mask = np.ones_like(y_true, dtype=bool)
        yt = y_true[mask]
        yp = y_pred[mask]
        return {
            'RMSE': np.sqrt(mean_squared_error(yt, yp)),
            'MAE': mean_absolute_error(yt, yp),
            'MedAE': median_absolute_error(yt, yp),
            'MAPE%': safe_mape(yt, yp),
            'SMAPE%': smape(yt, yp),
            'WMAPE%': wmape(yt, yp),
            'R2': r2_score(yt, yp) if len(np.unique(yt)) > 1 else np.nan
        }

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

    # 4.10 Προβλέψεις στο temp_fe (inference set)
    def predict_df(df: pd.DataFrame, model, features: list[str]) -> np.ndarray:
        df2 = df.copy()
        for col in features:
            if col not in df2.columns:
                df2[col] = 0
        X_pred = df2[features].fillna(0).to_numpy()
        y_hat = model.predict(X_pred)
        return np.maximum(y_hat, 0)

    for name, res in results.items():
        temp_fe[f'Predicted_Production_{name}'] = predict_df(temp_fe, res['model'], final_features)

    # 4.11 Έξοδος αρχείου με προβλέψεις
    out_cols = ['Date','Hour','Minute','Quarter','Temperature','Solar','Wind','Consumption'] + \
               [f'Predicted_Production_{n}' for n in results.keys()]
    out_df = temp_fe[out_cols].copy()
    out_path = 'predictions_temperature_hour_wind_consumption.xlsx'
    out_df.to_excel(out_path, index=False)
    print(f"[OK] Αποθηκεύτηκε: {out_path}")

    # 4.12 Διαγράμματα (προαιρετικά — χωρίς show για batch)
    plt.figure(figsize=(8, 6))
    sns.histplot(y_np, bins=30, kde=True, color='blue')
    plt.title('Distribution of Production'); plt.xlabel('Production'); plt.ylabel('Frequency'); plt.grid(True)

    plt.figure(figsize=(10, 8))
    corr_matrix = ermis_fe.select_dtypes(include=[np.number]).corr(numeric_only=True)
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Features')

    for name, res in results.items():
        plt.figure(figsize=(8, 6))
        train_sizes, train_scores, test_scores = learning_curve(
            res['model'], X_np, y_np, cv=5, scoring='r2', train_sizes=np.linspace(0.1, 1.0, 10)
        )
        plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training', color='blue')
        plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Validation', color='green')
        plt.title(f'Learning Curve – {name}')
        plt.xlabel('Training examples'); plt.ylabel('R²'); plt.legend(); plt.grid(True)

    for name, res in results.items():
        resid = y_test - res['y_pred']
        plt.figure(figsize=(8, 6))
        plt.scatter(range(len(resid)), resid, color='green', alpha=0.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.title(f'Residuals – {name}')
        plt.xlabel('Test Sample Index'); plt.ylabel('Actual - Predicted'); plt.grid(True)

    # Summary metrics table
    metrics_rows = []
    for name, res in results.items():
        row_all = {'Model': name, 'Split':'ALL', **res['full']}
        row_day = {'Model': name, 'Split':'DAY', **res['day']}
        row_nig = {'Model': name, 'Split':'NIGHT', **res['night']}
        metrics_rows.extend([row_all,row_day,row_nig])
    metrics_df = pd.DataFrame(metrics_rows)
    print("\n[SUMMARY METRICS]\n", metrics_df)

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



