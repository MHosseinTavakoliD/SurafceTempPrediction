import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
# Configuration
forecast_horizon = 6
look_back = 24
model_save_dir = './ModelForEachForecastHorizon/'
DataSource_file = 'C:/Users/zmx5fy/SurafceTempPrediction/Step4BuildingupDBforML/DBforMLaddingPredictionColAfterAfterCleaning/FinalDatasetForML6HourForecast.csv'
# Lists to store metrics for each model
mse_values = []
mae_values = []
# Ensure the directory for model saving exists
os.makedirs(model_save_dir, exist_ok=True)

# Load and preprocess data
df = pd.read_csv(DataSource_file)
df['MeasureTime'] = pd.to_datetime(df['MeasureTime'])

# Function to create the dataset for LightGBM for a specific target hour
def create_dataset_for_lightgbm(data, look_back=look_back, forecast_horizon=forecast_horizon):
    unique_stations = data['Station_name'].unique()
    X, Y = [], []

    for station in unique_stations:
        print (station)
        station_data = data[data['Station_name'] == station].copy()
        station_data.sort_values('MeasureTime', inplace=True)

        for i in range(len(station_data) - look_back - forecast_horizon + 1):
            past_features = station_data.iloc[i:i + look_back].drop(['MeasureTime', 'Station_name', 'County'], axis=1)
            target = station_data.iloc[i + look_back:i + look_back + forecast_horizon]['Surface TemperatureF']

            X.append(past_features.values.flatten())
            Y.append(target.values)

    return np.array(X), np.array(Y).reshape(-1, forecast_horizon)

# Create dataset
X, Y = create_dataset_for_lightgbm(df, look_back=look_back, forecast_horizon=forecast_horizon)

# Define a function for model training and evaluation
def train_and_evaluate(X_train, Y_train, X_val, Y_val, params):
    model = lgb.LGBMRegressor(
        objective=params['objective'],
        num_leaves=params['num_leaves'],
        learning_rate=params['learning_rate'],
        n_estimators=params['n_estimators']
    )

    # Create early stopping callback
    early_stopping = lgb.early_stopping(stopping_rounds=10, verbose=False)

    # Fit the model with early stopping callback
    model.fit(X_train, Y_train, eval_set=[(X_val, Y_val)], callbacks=[early_stopping])

    predictions = model.predict(X_val)
    mse = mean_squared_error(Y_val, predictions)
    mae = mean_absolute_error(Y_val, predictions)
    return mse, mae

# Hyperparameter grid
param_grid = {
    'objective': ['regression'],
    'num_leaves': [60, 70, 80, 90],
    'learning_rate': [0.1, 0.5, 1],
    'n_estimators': [100, 200, 500]
}

# Store the best parameters and corresponding MSE and MAE
best_params = None
best_mse = float('inf')
best_mae = float('inf')

# Iterate over each combination of parameters
for hour in range(forecast_horizon):
    Y_hour = Y[:, hour]  # Slice the target array for the specific hour
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y_hour, test_size=0.2, random_state=42)
    for objective in param_grid['objective']:
        for num_leaves in param_grid['num_leaves']:
            for learning_rate in param_grid['learning_rate']:
                for n_estimators in param_grid['n_estimators']:
                    params = {
                        'objective': objective,
                        'num_leaves': num_leaves,
                        'learning_rate': learning_rate,
                        'n_estimators': n_estimators
                    }

                    # Train and evaluate the model
                    mse, mae = train_and_evaluate(X_train, Y_train, X_val, Y_val, params)

                    # Update best params if current combination is better
                    if mse < best_mse:
                        best_mse = mse
                        best_mae = mae
                        best_params = params

                    print(f"hour: {hour} ,Params: {params}, MSE: {mse}, MAE: {mae}")

# Print the best parameters
print(f"Best Parameters: {best_params}, Best MSE: {best_mse}, Best MAE: {best_mae}")
# Best Parameters: {'objective': 'regression', 'num_leaves': 70, 'learning_rate': 0.1, 'n_estimators': 500},+
# Best MSE: 6.630115094374243, Best MAE: 1.446574885544757