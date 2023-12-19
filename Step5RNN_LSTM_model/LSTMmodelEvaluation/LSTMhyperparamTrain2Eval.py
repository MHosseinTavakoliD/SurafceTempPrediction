import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Your data preparation code goes here
# initial parameters:
Epoch = 150

forecast_horizon = 24
look_back = 24
# file_model_save = 'RNNLSTMV1HourForecast24Model2.h5'
DataSource_file = 'C:/Users/zmx5fy/SurafceTempPrediction/Step4BuildingupDBforML/DBforMLaddingPredictionColAfterAfterCleaning/FinalDatasetForML24HourForecast.csv'
# Check if TensorFlow is built with CUDA (GPU support)
print(tf.test.is_built_with_cuda())
tf.config.list_physical_devices('GPU')
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

            target = station_data.iloc[i + look_back:i + look_back + forecast_horizon]['Surface TemperatureF']

            X.append(past_features)
            Y.append(target.values)
            stations.append(station)

    return np.array(X), np.array(Y), np.array(stations)
# Create dataset
X, Y, station_names = create_dataset(df, look_back=look_back, forecast_horizon=forecast_horizon)
num_features = X.shape[2]
print ("Number of Features",X.shape[2])
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
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Define the LSTM model structures
# Hyperparameters
units_options = [50, 100]
dropout_options = [0.1, 0.2, 0.3]
batch_sizes = [8, 16, 32]
learning_rates = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
epoch = 100
# Define the LSTM model structures with hyperparameters
def define_model(model_type, input_shape, units, dropout_rate):
    model = Sequential()
    if model_type == "Stacked LSTM":
        model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units))
    elif model_type == "Bidirectional Deep LSTM":
        model.add(Bidirectional(LSTM(units, return_sequences=True), input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(units)))
    elif model_type == "Complex LSTM":
        model.add(Bidirectional(LSTM(units, return_sequences=True), input_shape=input_shape))
        model.add(LSTM(units))
        model.add(Dropout(dropout_rate))

    model.add(Dense(forecast_horizon))
    return model

model_types = ["Stacked LSTM", "Bidirectional Deep LSTM",  "Complex LSTM"]
# Initialize a list to store the results
model_results = []
history_dict = {}

for units in units_options:
    for dropout_rate in dropout_options:
        for batch_size in batch_sizes:
            for lr in learning_rates:
                for model_type in model_types:
                    print(
                        f"Training {model_type} with units={units}, dropout={dropout_rate}, batch_size={batch_size}, LR={lr}")

                    # Define and compile the model
                    model = define_model(model_type, sample_input_shape, units, dropout_rate)
                    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
                    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse', 'mae'])

                    # Train the model
                    history = model.fit(X_train, Y_train, epochs=epoch, batch_size=batch_size,
                                        validation_data=(X_val, Y_val), verbose=1)
                    # Evaluate the model on the validation set
                    predictions = model.predict(X_val)
                    val_mse = mean_squared_error(Y_val, predictions)
                    val_mae = mean_absolute_error(Y_val, predictions)
                    val_mape = mean_absolute_percentage_error(Y_val, predictions)
                    val_rmse = root_mean_squared_error(Y_val, predictions)

                    # Store the results
                    model_results.append({
                        "model_type": model_type,
                        "units": units,
                        "dropout_rate": dropout_rate,
                        "batch_size": batch_size,
                        "learning_rate": lr,
                        "min_val_loss": min(history.history['val_loss']),
                        "min_val_mse": val_mse,
                        "min_val_mae": val_mae,
                        "min_val_mape": val_mape,
                        "min_val_rmse": val_rmse
                    })
# Visualization code goes here - plot the metrics for each model

results_df = pd.DataFrame(model_results)
print(results_df)