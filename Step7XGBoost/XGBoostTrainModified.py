import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import joblib

# Constants
forecast_horizon = 24
look_back = 24
n_estimators = 250
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

# Create dataset
X, Y = create_dataset_for_xgboost(df, look_back=look_back, forecast_horizon=forecast_horizon)
print ("Number of Features",X.shape)
# Splitting the data
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define the XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror',
                         colsample_bytree=0.5,
                         learning_rate=0.1,
                         max_depth=5,
                         alpha=10,
                         n_estimators=n_estimators)

# Evaluation sets
eval_set = [(X_train, Y_train), (X_val, Y_val)]

# Train the model with evaluation metric
model.fit(X_train, Y_train, eval_metric=["rmse", "mae"], eval_set=eval_set, verbose=True, early_stopping_rounds=None)

# Plotting training progress
results = model.evals_result()
num_boost_rounds = len(results['validation_0']['rmse'])
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))

boosting_rounds = range(1, num_boost_rounds + 1)
ax1.plot(boosting_rounds, results['validation_0']['rmse'], label='Training RMSE')
ax1.plot(boosting_rounds, results['validation_1']['rmse'], label='Validation RMSE')
ax1.legend()
ax1.set_xlabel('Boosting Rounds')
ax1.set_ylabel('RMSE Value')
ax1.set_title('XGBoost RMSE')

ax2.plot(boosting_rounds, results['validation_0']['mae'], label='Training MAE')
ax2.plot(boosting_rounds, results['validation_1']['mae'], label='Validation MAE')
ax2.legend()
ax2.set_xlabel('Boosting Rounds')
ax2.set_ylabel('MAE Value')
ax2.set_title('XGBoost MAE')

plt.tight_layout()
plt.show()

# Evaluation with Eval_X_List and Eval_Y_List
from EvalList import Eval_X_List, Eval_Y_List

num_features = 117
for i in range(len(Eval_X_List)):
    X_val_sample = np.array(Eval_X_List[i])
    Y_val_sample = np.array(Eval_Y_List[i])

    # Make predictions
    predictions = model.predict(X_val_sample.reshape((-1, look_back * num_features)))
    actual_data = Y_val_sample.flatten()
    predicted_data = predictions.flatten()

    # Extract the entire sequence for the surface temperature feature
    past_surface_temp_sequence = X_val_sample[:, feature_index_for_surface_temp]

    plt.figure(figsize=(15, 6))
    plt.plot(range(-look_back, 0), past_surface_temp_sequence, label='Past Surface Temp', color='green')
    plt.plot(range(0, len(actual_data)), actual_data, label='Actual', color='blue')
    plt.plot(range(0, len(predicted_data)), predicted_data, label='Predicted', color='red')
    plt.title(f'Surface Temperature Prediction for Station Index: {i}')
    plt.xlabel('Time Steps (Relative to Prediction Point)')
    plt.ylabel('Temperature')
    plt.legend()
    plt.show()

    print("***************************")
    print(i)
    print(f"Past surface temp: {past_surface_temp_sequence}")
    print(f"Actual data: {actual_data}")
    print(f"Predicted data: {predicted_data}")