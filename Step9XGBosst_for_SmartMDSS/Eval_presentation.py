import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Constants
forecast_horizon = 24
look_back = 24
n_estimators = 250
feature_index_for_surface_temp = 2

# Load and preprocess data
DataSource_file = 'modified_DatasetForSmartMDSS.csv'
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

    for station in tqdm(unique_stations, desc='Processing Stations', unit=' station'):
        station_data = data[data['Station_name'] == station].copy()
        station_data.sort_values('MeasureTime', inplace=True)

        for i in range(len(station_data) - look_back - forecast_horizon + 1):
            current_time = station_data.iloc[i + look_back - 1]['MeasureTime']
            future_time = station_data.iloc[i + look_back]['MeasureTime']
            if (future_time - current_time).total_seconds() > 3600:
                continue

            past_features = station_data.iloc[i:i + look_back].drop(['MeasureTime', 'Station_name'], axis=1)
            target = station_data.iloc[i + look_back:i + look_back + forecast_horizon]['Surface TemperatureF']

            X.append(past_features.values.flatten())
            Y.append(target.values)

    return np.array(X), np.array(Y)

# Create dataset
X, Y = create_dataset_for_xgboost(df, look_back=look_back, forecast_horizon=forecast_horizon)
print("Number of Features", X.shape)
# Splitting the data
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Evaluation sets
eval_set = [(X_train, Y_train), (X_val, Y_val)]

# Load the saved model
model = joblib.load('xgboost_model.pkl')

# Define function to plot predictions for a single window
def plot_window_prediction(model, X_val_, Y_val_, feature_index_for_surface_temp, look_back):
    # Ensure X_val_ is 2-dimensional (should already be based on previous context)
    if X_val_.ndim == 1:
        X_val_ = X_val_.reshape(1, -1)

    # Make predictions
    predictions = model.predict(X_val_)
    actual_data = Y_val_.flatten()
    predicted_data = predictions.flatten()

    # Calculate the total number of features per timestep
    features_per_step = X_val_.shape[1] // look_back

    # Extract the past surface temperature sequence
    past_surface_temp_sequence = X_val_[0, feature_index_for_surface_temp::features_per_step]

    plt.figure(figsize=(15, 6))
    plt.plot(range(-look_back, 0), past_surface_temp_sequence, label='Past Surface Temp', color='green')
    plt.plot(range(0, len(actual_data)), actual_data, label='Actual', color='blue')
    plt.plot(range(0, len(predicted_data)), predicted_data, label='Predicted', color='red')
    plt.title('Surface Temperature Prediction')
    plt.xlabel('Time Steps (Relative to Prediction Point)')
    plt.ylabel('Temperature')
    plt.legend()
    plt.show()

# Select 10 random windows from the evaluation set and plot predictions
for i in range(10):
    X_val_sample = X_val[i]
    Y_val_sample = Y_val[i]
    plot_window_prediction(model, X_val_sample, Y_val_sample, feature_index_for_surface_temp, look_back)

