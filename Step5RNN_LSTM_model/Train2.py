import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from keras.layers import Bidirectional, BatchNormalization


# initial parameters:
Epoch = 250
LR = 0.0005
forecast_horizon = 24
look_back = 24
file_model_save = 'RNNLSTMV1HourForecast24Model2.h5'
DataSource_file = 'C:/Users/zmx5fy/SurafceTempPrediction/Step4BuildingupDBforML/DBforMLaddingPredictionColAfterAfterCleaning/FinalDatasetForML24HourForecast.csv'
# Check if TensorFlow is built with CUDA (GPU support)
print(tf.test.is_built_with_cuda())

# Check available GPUs in the system
print(tf.config.list_physical_devices('GPU'))

# Load and preprocess data
df = pd.read_csv(DataSource_file)
df['MeasureTime'] = pd.to_datetime(df['MeasureTime'])
# Extract time components
df['hour'] = df['MeasureTime'].dt.hour
df['day_of_week'] = df['MeasureTime'].dt.dayofweek
df['day_of_month'] = df['MeasureTime'].dt.day
df['month'] = df['MeasureTime'].dt.month
df['year'] = df['MeasureTime'].dt.year

# Encode cyclical features
def encode_cyclical_feature(df, col, max_vals):
    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_vals)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_vals)
    return df

df = encode_cyclical_feature(df, 'hour', 24)
df = encode_cyclical_feature(df, 'day_of_week', 7)
df = encode_cyclical_feature(df, 'month', 12)
# Function to create dataset
def create_dataset(data, look_back=24, forecast_horizon=12):
    unique_stations = data['Station_name'].unique()
    X, Y, stations = [], [], []

    for station in unique_stations:
        station_data = data[data['Station_name'] == station].copy()
        station_data.sort_values('MeasureTime', inplace=True)

        for i in range(len(station_data) - look_back - forecast_horizon + 1):
            current_time = station_data.iloc[i + look_back - 1]['MeasureTime']
            future_time = station_data.iloc[i + look_back]['MeasureTime']
            if (future_time - current_time).total_seconds() > 3600:
                continue
            # Historical features

            past_features = station_data.iloc[i:i + look_back].drop(['MeasureTime', 'Station_name',  'County'], axis=1) #'Surface TemperatureF',

            # Future forecasted features
            # future_features = station_data.iloc[i + look_back:i + look_back + forecast_horizon][['Air TemperatureF', 'Rel. Humidity%', 'Wind Speed (act)mph', 'Precipitation Intensityin/h']]
            # Flatten past_features to a 1D array
            # past_features_flat = past_features.values.flatten()

            # Flatten future_features to a 1D array
            # future_features_flat = future_features.values.flatten()

            # Combine past and future features
            # combined_features = np.concatenate((past_features_flat, future_features_flat))

            # Reshape combined_features to 2D array for LSTM input
            # combined_features = combined_features.reshape(1, -1)

            target = station_data.iloc[i + look_back:i + look_back + forecast_horizon]['Surface TemperatureF']

            X.append(past_features)
            Y.append(target.values)
            stations.append(station)

    return np.array(X), np.array(Y), np.array(stations)

# Create dataset
X, Y, station_names = create_dataset(df, look_back=look_back, forecast_horizon=forecast_horizon)

# Splitting the data for each station
train_X, val_X, train_Y, val_Y = {}, {}, {}, {}
for station in np.unique(station_names):
    idx = station_names == station
    X_train, X_val, Y_train, Y_val = train_test_split(X[idx], Y[idx], test_size=0.2, random_state=42)
    train_X[station] = X_train
    val_X[station] = X_val
    train_Y[station] = Y_train
    val_Y[station] = Y_val

print (X_val[0])
print (Y_val[0])
# LSTM Model Construction model 1
sample_input_shape = next(iter(train_X.values())).shape[1:]
# LSTM Model Construction model 1
# model = Sequential()
# model.add(LSTM(50, return_sequences=True, input_shape=sample_input_shape))
# model.add(Dropout(0.2))
# model.add(LSTM(50))
# model.add(Dropout(0.2))
# model.add(Dense(forecast_horizon))

# LSTM Model Construction model 2
model = Sequential()
model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=sample_input_shape))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(100, return_sequences=True)))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(100)))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(forecast_horizon))  # Assuming your output size is 12


optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse', 'mae'])

# Model Training
history = model.fit(X_train, Y_train, epochs=Epoch, batch_size=32, validation_data=(X_val, Y_val), verbose=1)

# After training the model
model.save(file_model_save)

# Visualization of Metrics
plt.figure(figsize=(12, 4))

# Plot training & validation loss values
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation MSE
plt.subplot(1, 3, 2)
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('Model MSE')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation MAE
plt.subplot(1, 3, 3)
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()
# Model Evaluation
predictions = model.predict(X_val)
val_mse = mean_squared_error(Y_val, predictions)
val_mae = mean_absolute_error(Y_val, predictions)

# # Visualization
# random_index = np.random.randint(0, len(Y_val))
# actual_data = Y_val[random_index].flatten()
# predicted_data = predictions[random_index].flatten()
#
# plt.figure(figsize=(10, 6))
# plt.plot(actual_data, label='Actual', color='blue')
# plt.plot(predicted_data, label='Predicted', color='red')
# plt.title(f'Surface Temperature Prediction for Random Station Index: {random_index}')
# plt.xlabel('Time Steps')
# plt.ylabel('Temperature')
# plt.legend()
# plt.show()
# Visualization
list_index = [134, 462, 368, 529, 453, 375]
feature_index_for_surface_temp = 2


# Remove the last 48 elements from each sample
# X_val_trimmed = np.array([sample[0, :-48] for sample in X_val])

# Reshape X_val_trimmed to have a shape of (number of samples, look_back, num_features)
# X_val_reshaped = X_val.reshape(X_val.shape[0], look_back, num_features)

for i in range(len(list_index)):
    random_index = list_index[i]

    actual_data = Y_val[random_index].flatten()
    predicted_data = predictions[random_index].flatten()

    past_surface_temp = X_val[random_index, :, feature_index_for_surface_temp]

    plt.figure(figsize=(15, 6))

    # Plot past surface temperature
    plt.plot(range(-len(past_surface_temp), 0), past_surface_temp, label='Past Surface Temp', color='green')

    # Plot actual surface temperature for the forecast horizon
    plt.plot(range(0, len(actual_data)), actual_data, label='Actual', color='blue')

    # Plot predicted surface temperature for the forecast horizon
    plt.plot(range(0, len(predicted_data)), predicted_data, label='Predicted', color='red')

    plt.title(f'Surface Temperature Prediction for Station Index: {random_index}')
    plt.xlabel('Time Steps (Relative to Prediction Point)')
    plt.ylabel('Temperature')
    plt.legend()
    plt.show()
