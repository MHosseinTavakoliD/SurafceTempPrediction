import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
# Constants
forecast_horizon = 24
look_back = 24
n_estimators = 20
feature_index_for_surface_temp = 2

# Load and preprocess data
DataSource_file = 'C:/Users/zmx5fy/SurafceTempPrediction/Step4BuildingupDBforML/DBforMLaddingPredictionColAfterAfterCleaning/FinalDatasetForML24HourForecast.csv'
df = pd.read_csv(DataSource_file)
df['MeasureTime'] = pd.to_datetime(df['MeasureTime'])

# Extract time components and encode cyclical features
def encode_cyclical_feature(df, col, max_vals):
    df[col + '_sin'] = np.sin(2 * np.pi * df[col] / max_vals)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col] / max_vals)
    return df

df['hour'] = df['MeasureTime'].dt.hour
df['day_of_week'] = df['MeasureTime'].dt.dayofweek
df['day_of_month'] = df['MeasureTime'].dt.day
df['month'] = df['MeasureTime'].dt.month
df['year'] = df['MeasureTime'].dt.year
df = encode_cyclical_feature(df, 'hour', 24)
df = encode_cyclical_feature(df, 'day_of_week', 7)
df = encode_cyclical_feature(df, 'month', 12)


# Define function to create dataset for XGBoost
def create_dataset_for_xgboost(data, look_back=look_back, forecast_horizon=forecast_horizon):
    unique_stations = data['Station_name'].unique()
    X, Y = [], []

    for station in unique_stations:
        station_data = data[data['Station_name'] == station].copy()
        station_data.sort_values('MeasureTime', inplace=True)

        for i in range(len(station_data) - look_back - forecast_horizon + 1):
            current_time = station_data.iloc[i + look_back - 1]['MeasureTime']
            future_time = station_data.iloc[i + look_back]['MeasureTime']
            if (future_time - current_time).total_seconds() > 3600:
                continue

            past_features = station_data.iloc[i:i + look_back].drop(['MeasureTime', 'Station_name', 'County'], axis=1)
            target = station_data.iloc[i + look_back:i + look_back + forecast_horizon]['Surface TemperatureF']

            X.append(past_features.values.flatten())
            Y.append(target.values)

    return np.array(X), np.array(Y)

# Define your custom loss function here
def custom_loss_function(y_true, y_pred):
    # Example loss function: mean squared error
    return np.mean((y_true - y_pred) ** 2)

# Create dataset
X, Y = create_dataset_for_xgboost(df, look_back=look_back, forecast_horizon=forecast_horizon)

# Splitting the data
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.3, 0.5, 0.9],
    'colsample_bytree': [0.3, 0.5, 0.9]
}

# Initialize XGBoost regressor
xgb_model = xgb.XGBRegressor(objective='reg:squarederror')

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_grid,
                                   n_iter=10, scoring='neg_mean_squared_error', cv=3,
                                   verbose=1, n_jobs=1, random_state=42)

# Fit the model
random_search.fit(X_train, Y_train)

# Extract the top models from the grid search
top_indices = np.argsort(random_search.cv_results_['mean_test_score'])[-3:][::-1]
top_model_params = [random_search.cv_results_['params'][i] for i in top_indices]

# Evaluate top models on MSE, MAE, and custom loss
model_evaluations = []
for params in top_model_params:
    model = xgb.XGBRegressor(**params, objective='reg:squarederror')
    model.fit(X_train, Y_train)

    predictions = model.predict(X_val)
    mse = mean_squared_error(Y_val, predictions)
    mae = mean_absolute_error(Y_val, predictions)
    custom_loss_value = custom_loss_function(Y_val, predictions)

    model_evaluations.append({
        'params': params,
        'mse': mse,
        'mae': mae,
        'custom_loss': custom_loss_value
    })
pd.set_option('display.max_rows', 500)  # or any large number like 500
pd.set_option('display.max_columns', 10)  # or the number of columns you have
# Display results
for i, evaluation in enumerate(model_evaluations, 1):
    print(f"Model {i}:")
    print(f"Parameters: {evaluation['params']}")
    print(f"MSE: {evaluation['mse']}, MAE: {evaluation['mae']}, Custom Loss: {evaluation['custom_loss']}")
    print("---------------------------------------------------")