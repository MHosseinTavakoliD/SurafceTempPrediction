import pandas as pd

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



df = pd.read_csv('C:/Users/zmx5fy/SurafceTempPrediction/Step4BuildingupDBforML/MonthlyGraphsHourlyAfterCleaning/AfterProcessedHourly_dataset.csv')


def create_dataset(data, look_back=24, forecast_horizon=12):
    unique_stations = data['Station_name'].unique()
    X, Y, stations = [], [], []

    for station in unique_stations:
        station_data = data[data['Station_name'] == station]
        station_data.sort_values('MeasureTime', inplace=True)

        # Loop through the data
        for i in range(len(station_data)):
            if i + look_back + forecast_horizon > len(station_data):
                break

            # Check for a gap in the data
            current_time = station_data.iloc[i]['MeasureTime']
            future_time = station_data.iloc[i + look_back]['MeasureTime']
            if (future_time - current_time).total_seconds() > (look_back * 3600):
                continue  # Skip this window due to gap

            # Extract features and labels
            X.append(station_data.iloc[i:i + look_back].drop(['MeasureTime', 'Station_name', 'Air TemperatureF', 'Precipitation Intensityin/h'], axis=1).to_numpy())
            Y.append(station_data.iloc[i + look_back:i + look_back + forecast_horizon][['Air TemperatureF', 'Precipitation Intensityin/h']].to_numpy())
            stations.append(station)

    return np.array(X), np.array(Y), np.array(stations)

# Usage

df['MeasureTime'] = pd.to_datetime(df['MeasureTime'])

look_back = 24
forecast_horizon = 12
X, Y, station_names = create_dataset(df, look_back, forecast_horizon)
print ("X", X.shape)
print ("Y", Y.shape)

# Splitting the data for each station
train_X, val_X, train_Y, val_Y = {}, {}, {}, {}

for station in np.unique(station_names):
    idx = station_names == station
    X_train, X_val, Y_train, Y_val = train_test_split(X[idx], Y[idx], test_size=0.2, random_state=42)
    train_X[station] = X_train
    val_X[station] = X_val
    train_Y[station] = Y_train
    val_Y[station] = Y_val
