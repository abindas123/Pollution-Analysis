import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load dataset with correct delimiter
file_path = 'C:\\Users\\Abindas\\Downloads\\Assignment 2\\AirQuality.csv'

# Specify the correct delimiter (semicolon) when reading the CSV file
try:
    df = pd.read_csv(file_path, delimiter=';', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(file_path, delimiter=';', encoding='ISO-8859-1')

# Print column names BEFORE renaming
print("Columns before renaming:", df.columns.tolist())

# Standardize column names (replace spaces with underscores and remove periods)
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('.', '')

# Print column names AFTER renaming
print("Columns after renaming:", df.columns.tolist())

# Check for missing columns
required_cols = {'PM2_5', 'PM10', 'AQI'}
missing_cols = required_cols - set(df.columns)

if missing_cols:
    print(f"🚨 Error: Missing columns - {missing_cols}")
    print("Possible column names:", df.columns.tolist())
    # You can decide to either exit or proceed without those columns
    # exit()  # Uncomment if you want to stop execution
else:
    print("✅ All required columns found!")

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')

# Check if any date values could not be parsed
print(f"Number of missing dates after conversion: {df['Date'].isnull().sum()}")

# Feature Engineering - Creating Time-based Features
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day

# Replace commas with periods for columns that should be numeric
numeric_columns = ['CO(GT)', 'PT08S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08S2(NMHC)', 'NOx(GT)', 
                   'PT08S3(NOx)', 'NO2(GT)', 'PT08S4(NO2)', 'PT08S5(O3)', 'T', 'RH', 'AH']

# Iterate over the numeric columns and replace commas with periods, then convert to numeric
for col in numeric_columns:
    if col in df.columns:
        df[col] = df[col].replace({',': '.'}, regex=True)  # Replace commas with periods
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, coercing errors to NaN

# Print the column names to debug the issue
print("Column names in dataset after renaming:", df.columns.tolist())

# Selecting Features and Target
features = ['CO(GT)', 'PT08S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08S2(NMHC)', 'NOx(GT)', 
            'PT08S3(NOx)', 'NO2(GT)', 'PT08S4(NO2)', 'PT08S5(O3)', 'T', 'RH', 'AH', 'year', 'month', 'day']

target = 'CO(GT)'  # Change target to any available column (e.g., 'CO(GT)')

# Ensure the columns exist in the dataset
for feature in features:
    if feature not in df.columns:
        print(f"⚠️ Warning: Column '{feature}' not found. Skipping it.")
        
X = df[features]
y = df[target]

# Handling missing values
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)
y = imputer.fit_transform(y.values.reshape(-1, 1))  # If target has missing values

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Data Visualization
plt.figure(figsize=(10,5))
sns.histplot(df['CO(GT)'], bins=30, kde=True, color='blue')
plt.title('Distribution of CO Levels')
plt.xlabel('CO')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(12,6))
sns.lineplot(x=df['Date'], y=df['NO2(GT)'], label='NO2', color='red')
sns.lineplot(x=df['Date'], y=df['CO(GT)'], label='CO', color='blue')
plt.title('Trend of NO2 and CO Over Time')
plt.xlabel('Date')
plt.ylabel('Pollutant Concentration')
plt.legend()
plt.show()

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

# Evaluation - Random Forest
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_r2 = r2_score(y_test, y_pred_rf)
print(f'Random Forest RMSE: {rf_rmse}, R2: {rf_r2}')

# LSTM Model
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, validation_data=(X_test_lstm, y_test))

y_pred_lstm = model.predict(X_test_lstm).flatten()  # Flatten to match y_test shape

# Evaluation - LSTM
lstm_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lstm))
lstm_r2 = r2_score(y_test, y_pred_lstm)
print(f'LSTM RMSE: {lstm_rmse}, R2: {lstm_r2}')

# Exporting Predictions for Tableau
results = pd.DataFrame({'Actual': y_test.flatten(), 'RF_Predicted': y_pred_rf, 'LSTM_Predicted': y_pred_lstm})
results.to_csv('AirQuality_Predictions.csv', index=False)
