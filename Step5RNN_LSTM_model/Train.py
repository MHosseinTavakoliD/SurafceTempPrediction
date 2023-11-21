import pandas as pd

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



df = pd.read_csv('C:/Users/zmx5fy/SurafceTempPrediction/Step4BuildingupDBforML/MonthlyGraphsHourlyAfterCleaning/AfterProcessedHourly_dataset.csv')
# Check for string columns in the dataset
print(df.dtypes)

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


# 2. LSTM Model Construction
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Assuming train_X is a dictionary containing training data for each station
# We need to get the shape of the input from any station's data
sample_input_shape = next(iter(train_X.values())).shape[1:]  # Gets the input shape from the first station's data

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=sample_input_shape))
model.add(Dropout(0.2))  # Adding dropout for regularization
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(24))  # Adjust the output layer according to your forecast_horizon and output features

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Model summary (optional, to verify the model structure)
model.summary()


# Convert data to float32
X_train = X_train.astype('float32')
Y_train = Y_train.astype('float32')

# Checking for NaN and infinite values and handling them
import numpy as np

if np.isnan(X_train).any() or np.isinf(X_train).any():
    # Handle or remove NaNs and infinite values
    X_train = np.nan_to_num(X_train)

if np.isnan(Y_train).any() or np.isinf(Y_train).any():
    # Handle or remove NaNs and infinite values
    Y_train = np.nan_to_num(Y_train)


# 3. Model Training
# Train the model
model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=(X_val, Y_val), verbose=1)
# 4. Model Evaluation

# Make predictions
predictions = model.predict(X_val)

# Inverse transform predictions and actual values to compare
predictions_inverse = scaler.inverse_transform(predictions)
Y_val_inverse = scaler.inverse_transform(Y_val)

# Calculate performance metrics as needed (e.g., MSE, RMSE)