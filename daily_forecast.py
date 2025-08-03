from datetime import datetime, time, timedelta
from energyquantified import EnergyQuantified
from energyquantified.time import Frequency
import pandas as pd
import numpy as np
import os
import pytz
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor


# ✅ --- EnergyQuantified API Key από τα secrets ---
eq = EnergyQuantified(api_key=os.getenv("EQ_API_KEY"))

# Ημερομηνίες
today = datetime.now().date()
tomorrow = today + timedelta(days=1)
day_after_tomorrow = today + timedelta(days=2)
begin = datetime.combine(today, time(0, 0))
end = datetime.combine(day_after_tomorrow, time(0, 0))

# Συνάρτηση για φόρτωση προβλέψεων χωρίς επαναλήψεις
def load_forecast_both_days(series_name, col_name):
    all_data = []
    for days_ahead in [0, 1]:
        instance = eq.instances.relative(
            series_name,
            begin=begin,
            end=end,
            tag='ec',
            days_ahead=days_ahead,
            time_of_day=time(0, 0),
            frequency=Frequency.PT1H
        )
        df = pd.DataFrame(instance.data)
        if df.empty:
            continue
        df['date'] = pd.to_datetime(df['date'])

        # Φιλτράρισμα για αποφυγή επικαλύψεων
        if days_ahead == 0:
            df = df[df['date'].dt.date == today]
        elif days_ahead == 1:
            df = df[df['date'].dt.date == tomorrow]

        df = df.set_index('date')
        df = df.rename(columns={'value': col_name})
        all_data.append(df)
    return pd.concat(all_data)

# Φόρτωση όλων των προβλέψεων
temperature = load_forecast_both_days('GR Consumption Temperature °C 15min Forecast', 'Temperature')
solar = load_forecast_both_days('GR Solar Photovoltaic Production MWh/h 15min Forecast', 'Solar')
wind = load_forecast_both_days('GR Wind Power Production MWh/h 15min Forecast', 'Wind')
load = load_forecast_both_days('GR Consumption MWh/h 15min Forecast', 'Consumption')

# Συνένωση
concat = pd.concat([temperature, solar, wind, load], axis=1).reset_index()

# Δημιουργία Date και Hour
concat['Date'] = concat['date'].dt.date
concat['Hour'] = concat['date'].dt.hour

# Τελική μορφοποίηση
concat = concat[['Date', 'Hour', 'Temperature', 'Solar', 'Wind', 'Consumption']]

# Αποθήκευση
concat.to_csv('temp1.csv', sep=';', index=False)
print(concat.head())


# Load the ERMIS Excel file (for training)
ermis_file_path = 'ERMIS_II - Real1.xlsx'
ermis_df = pd.read_excel(ermis_file_path, sheet_name='Sheet1')

# Strip any leading/trailing spaces in the column names
ermis_df.rename(columns=lambda x: x.strip(), inplace=True)

# Load the temperature and solar forecast file (for prediction)
temp_file_path = 'temp1.csv'
temp_df = pd.read_csv(temp_file_path, delimiter=';')

# Ensure required columns exist in temp_df
missing_columns = [col for col in ['Temperature', 'Solar', 'Wind', 'Consumption'] if col not in temp_df.columns]
if missing_columns:
    raise KeyError(f"Οι ακόλουθες στήλες λείπουν από το temp_df: {missing_columns}")

# Convert date format
temp_df['Date'] = pd.to_datetime(temp_df['Date'], format='%Y-%m-%d')
ermis_df['Date'] = pd.to_datetime(ermis_df['Date'], format='%Y-%m-%d')
# WEEKDAY
temp_df['Weekday'] = temp_df['Date'].dt.weekday
ermis_df['Weekday'] = ermis_df['Date'].dt.weekday

# IS_WEEKEND (binary flag)
temp_df['Is_Weekend'] = temp_df['Weekday'].apply(lambda x: 1 if x >= 5 else 0)
ermis_df['Is_Weekend'] = ermis_df['Weekday'].apply(lambda x: 1 if x >= 5 else 0)

# MONTH
temp_df['Month'] = temp_df['Date'].dt.month
ermis_df['Month'] = ermis_df['Date'].dt.month

# SEASON (based on month)
def season(month):
    if month in [12,1,2]:
        return 0  # Winter
    elif month in [3,4,5]:
        return 1  # Spring
    elif month in [6,7,8]:
        return 2  # Summer
    else:
        return 3  # Fall

temp_df['Season'] = temp_df['Month'].apply(season)
ermis_df['Season'] = ermis_df['Month'].apply(season)

# HOUR BIN
def hour_bin(hour):
    if 0 <= hour < 6:
        return 0  # Night
    elif 6 <= hour < 12:
        return 1  # Morning
    elif 12 <= hour < 18:
        return 2  # Afternoon
    else:
        return 3  # Evening

temp_df['Hour_Bin'] = temp_df['Hour'].apply(hour_bin)
ermis_df['Hour_Bin'] = ermis_df['Hour'].apply(hour_bin)

# TEMP BIN
def temp_bin(temp):
    if temp < 10:
        return 0
    elif temp < 20:
        return 1
    else:
        return 2

temp_df['Temp_Bin'] = temp_df['Temperature'].apply(temp_bin)
ermis_df['Temp_Bin'] = ermis_df['Temperature'].apply(temp_bin)

# INTERACTION FEATURES
temp_df['Solar_Temp'] = temp_df['Solar'] * temp_df['Temperature']
ermis_df['Solar_Temp'] = ermis_df['Solar'] * ermis_df['Temperature']

temp_df['Wind_Temp'] = temp_df['Wind'] * temp_df['Temperature']
ermis_df['Wind_Temp'] = ermis_df['Wind'] * ermis_df['Temperature']

temp_df['Cons_Temp'] = temp_df['Consumption'] * temp_df['Temperature']
ermis_df['Cons_Temp'] = ermis_df['Consumption'] * ermis_df['Temperature']

temp_df['Hour_Solar'] = temp_df['Hour'] * temp_df['Solar']
ermis_df['Hour_Solar'] = ermis_df['Hour'] * ermis_df['Solar']

temp_df['Weekday_Consumption'] = temp_df['Weekday'] * temp_df['Consumption']
ermis_df['Weekday_Consumption'] = ermis_df['Weekday'] * ermis_df['Consumption']

# SIN/COS ENCODING OF HOUR
temp_df['Hour_sin'] = np.sin(2 * np.pi * temp_df['Hour'] / 24)
temp_df['Hour_cos'] = np.cos(2 * np.pi * temp_df['Hour'] / 24)
ermis_df['Hour_sin'] = np.sin(2 * np.pi * ermis_df['Hour'] / 24)
ermis_df['Hour_cos'] = np.cos(2 * np.pi * ermis_df['Hour'] / 24)

# NORMALIZED CONSUMPTION
max_consumption = ermis_df['Consumption'].max()
temp_df['Normalized_Consumption'] = temp_df['Consumption'] / max_consumption
ermis_df['Normalized_Consumption'] = ermis_df['Consumption'] / max_consumption

# LAG FEATURES (only for ERMIS, because temp_df lacks history)
for lag in [1, 2, 3]:
    ermis_df[f'Consumption_lag{lag}'] = ermis_df['Consumption'].shift(lag)
    ermis_df[f'Solar_lag{lag}'] = ermis_df['Solar'].shift(lag)
    ermis_df[f'Wind_lag{lag}'] = ermis_df['Wind'].shift(lag)

# ROLLING FEATURES
ermis_df['Consumption_roll3_mean'] = ermis_df['Consumption'].rolling(window=3).mean()
ermis_df['Consumption_roll3_std'] = ermis_df['Consumption'].rolling(window=3).std()


# --- Δείκτες Load / Solar / Wind Ratio ---
ermis_df['Load_Solar_Ratio'] = ermis_df['Consumption'] / (ermis_df['Solar'] + 1)
temp_df['Load_Solar_Ratio'] = temp_df['Consumption'] / (temp_df['Solar'] + 1)

ermis_df['Load_Wind_Ratio'] = ermis_df['Consumption'] / (ermis_df['Wind'] + 1)
temp_df['Load_Wind_Ratio'] = temp_df['Consumption'] / (temp_df['Wind'] + 1)

# --- Σχετική ηλιακή παραγωγή ---
ermis_df['Solar_Relative'] = ermis_df['Solar'] / (ermis_df.groupby('Date')['Solar'].transform('max') + 1)
temp_df['Solar_Relative'] = temp_df['Solar'] / (temp_df.groupby('Date')['Solar'].transform('max') + 1)

# --- Interaction season * hour ---
ermis_df['Season_Hour'] = ermis_df['Season'] * ermis_df['Hour']
temp_df['Season_Hour'] = temp_df['Season'] * temp_df['Hour']

# --- Is Peak Hour ---
ermis_df['Is_Peak_Hour'] = ermis_df['Hour'].apply(lambda x: 1 if 11 <= x <= 17 else 0)
temp_df['Is_Peak_Hour'] = temp_df['Hour'].apply(lambda x: 1 if 11 <= x <= 17 else 0)

# --- Daily max/min consumption ---
ermis_df['Daily_Max_Consumption'] = ermis_df.groupby('Date')['Consumption'].transform('max')
ermis_df['Daily_Min_Consumption'] = ermis_df.groupby('Date')['Consumption'].transform('min')

temp_df['Daily_Max_Consumption'] = temp_df.groupby('Date')['Consumption'].transform('max')
temp_df['Daily_Min_Consumption'] = temp_df.groupby('Date')['Consumption'].transform('min')

# --- Polynomial features ---
ermis_df['Temperature_sq'] = ermis_df['Temperature'] ** 2
temp_df['Temperature_sq'] = temp_df['Temperature'] ** 2

ermis_df['Consumption_sq'] = ermis_df['Consumption'] ** 2
temp_df['Consumption_sq'] = temp_df['Consumption'] ** 2



# FILL NA
ermis_df.fillna(0, inplace=True)



# Convert numeric columns from string to float
def convert_to_float(df, column_name):
    if column_name in df.columns:
        try:
            df[column_name] = df[column_name].astype(str).str.replace(',', '.').astype(float)
        except Exception as e:
            raise ValueError(f"Σφάλμα στη μετατροπή της στήλης {column_name} σε float: {e}")

for col in ['Temperature', 'Solar', 'Wind', 'Consumption']:
    convert_to_float(temp_df, col)

# Ensure the 'Hour' column is correctly formatted as integer
temp_df['Hour'] = temp_df['Hour'].astype(int)

# Create the 'Ordinal_Date' column
temp_df['Ordinal_Date'] = temp_df['Date'].map(pd.Timestamp.toordinal)
ermis_df['Ordinal_Date'] = ermis_df['Date'].map(pd.Timestamp.toordinal)

required_columns = [
    'Ordinal_Date', 'Weekday', 'Is_Weekend', 'Month', 'Season', 'Hour', 'Hour_Bin',
    'Temperature', 'Temp_Bin', 'Solar', 'Wind', 'Consumption',
    'Solar_Temp', 'Wind_Temp', 'Cons_Temp',
    'Hour_Solar', 'Weekday_Consumption',
    'Hour_sin', 'Hour_cos',
    'Normalized_Consumption',
    'Consumption_lag1', 'Consumption_lag2', 'Consumption_lag3',
    'Solar_lag1', 'Solar_lag2', 'Solar_lag3',
    'Wind_lag1', 'Wind_lag2', 'Wind_lag3',
    'Consumption_roll3_mean', 'Consumption_roll3_std','Load_Solar_Ratio', 'Load_Wind_Ratio',
    'Solar_Relative',
    'Season_Hour',
    'Is_Peak_Hour',
    'Daily_Max_Consumption', 'Daily_Min_Consumption',
    'Temperature_sq', 'Consumption_sq'
]



available_columns = [col for col in required_columns if col in ermis_df.columns]
X = ermis_df[available_columns]
y = ermis_df['Production']  # Target: Production

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle NaN values
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)
# Κρατάμε το schema των features που είδε το μοντέλο
final_features = X_train.columns.tolist()

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

# Train models
models = {}
for name, model, param_grid in zip(
    ['RandomForest', 'GradientBoosting', 'XGBoost'],
    [RandomForestRegressor(random_state=42), GradientBoostingRegressor(random_state=42), XGBRegressor(random_state=42)],
    [param_grid_rf, param_grid_gb, param_grid_xgb]
):
    print(f"Starting GridSearch for {name}...")
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)
    print(f"Best parameters for {name}: {grid_search.best_params_}")
    models[name] = grid_search.best_estimator_

# Make predictions
results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = np.maximum(model.predict(X_test), 0)
    results[name] = {
        'model': model,
        'mse': mean_squared_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'y_pred': y_pred
    }

def predict_24_hours_from_file(df, model, final_features):
    for col in final_features:
        if col not in df.columns:
            df[col] = 0
    X_pred = df[final_features]
    return np.maximum(model.predict(X_pred), 0)



for name, result in results.items():
    print(f"Predictions for {name}:")
    temp_df[f'Predicted_Production_{name}'] = predict_24_hours_from_file(temp_df, result['model'], final_features)

# Save predictions to file
output_path = 'predictions_temperature_hour_wind_consumption.xlsx'
temp_df.to_excel(output_path, index=False)
print(f"Predictions saved to: {output_path}")

# Plot Data Distribution
plt.figure(figsize=(8,6))
sns.histplot(y, bins=30, kde=True, color='blue')
plt.title('Distribution of Production')
plt.xlabel('Production')
plt.ylabel('Frequency')
plt.grid(True)
#plt.show()

# Correlation Matrix
plt.figure(figsize=(8,6))
corr_matrix = ermis_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Features')
#plt.show()

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = np.maximum(model.predict(X_test), 0)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"{name} Metrics:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared: {r2:.4f}\n")
    
    results[name] = {
        'model': model,
        'mse': mse,
        'r2': r2,
        'y_pred': y_pred
    }



# Learning Curves
for name, result in results.items():
    plt.figure(figsize=(8,6))
    train_sizes, train_scores, test_scores = learning_curve(result['model'], X, y, cv=5, scoring='r2', train_sizes=np.linspace(0.1, 1.0, 10))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score', color='blue')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Validation score', color='green')
    plt.title(f'Learning Curve for {name}')
    plt.xlabel('Training examples')
    plt.ylabel('R² Score')
    plt.legend()
    plt.grid(True)
    #plt.show()


# Residuals Plot
for name, result in results.items():
    residuals = y_test - result['y_pred']
    plt.figure(figsize=(8,6))
    plt.scatter(range(len(y_test)), residuals, color='green', alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'Residuals Plot for {name}')
    plt.xlabel('Test Sample Index')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.grid(True)
    #plt.show()
    

print(temp_df.tail(3))


