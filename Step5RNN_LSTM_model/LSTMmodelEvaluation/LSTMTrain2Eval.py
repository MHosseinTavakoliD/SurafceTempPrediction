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
LR = 0.0001
forecast_horizon = 24
look_back = 24
# file_model_save = 'RNNLSTMV1HourForecast24Model2.h5'
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
# def mean_absolute_percentage_error(y_true, y_pred):
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#
# def root_mean_squared_error(y_true, y_pred):
#     return np.sqrt(mean_squared_error(y_true, y_pred))

# Define the LSTM model structures
def define_model(model_type, input_shape):
    model = Sequential()
    if model_type == "Simple LSTM":
        model.add(LSTM(100, input_shape=input_shape))
    elif model_type == "Stacked LSTM":
        model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(100))
    elif model_type == "Deep LSTM":
        model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(100, return_sequences=True))
        model.add(LSTM(100))
    elif model_type == "Bidirectional LSTM":
        model.add(Bidirectional(LSTM(100), input_shape=input_shape))
    elif model_type == "LSTM with Dropout":
        model.add(LSTM(100, input_shape=input_shape))
        model.add(Dropout(0.1))
    elif model_type == "Wide LSTM":
        model.add(LSTM(100, input_shape=input_shape))
    elif model_type == "Deep and Wide LSTM":
        model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(100))
    elif model_type == "Bidirectional Deep LSTM":
        model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=input_shape))
        model.add(Bidirectional(LSTM(100)))
    elif model_type == "LSTM with Batch Normalization":
        model.add(LSTM(100, input_shape=input_shape))
        model.add(BatchNormalization())
    elif model_type == "Complex LSTM":
        model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=input_shape))
        model.add(LSTM(100))
        model.add(Dropout(0.1))

    model.add(Dense(forecast_horizon))
    return model

model_types = ["Simple LSTM", "Stacked LSTM", "Deep LSTM", "Bidirectional LSTM",
               "LSTM with Dropout", "Wide LSTM", "Deep and Wide LSTM",
               "Bidirectional Deep LSTM", "LSTM with Batch Normalization", "Complex LSTM"]

history_dict = {}

for model_type in model_types:
    print(f"Training model: {model_type}")
    model = define_model(model_type, sample_input_shape)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse', 'mae'])
    history = model.fit(X_train, Y_train, epochs=Epoch, batch_size=32, validation_data=(X_val, Y_val), verbose=1)
    history_dict[model_type] = history

# Visualization code goes here - plot the metrics for each model


# Number of models
num_models = len(history_dict)
model_names = list(history_dict.keys())
# Plot Loss for each model
plt.figure(figsize=(10, 5))  # Adjust the size as needed
for model_name, history in history_dict.items():
    plt.plot(history.history['loss'], label=f'{model_name} - Train Loss')
    plt.plot(history.history['val_loss'], label=f'{model_name} - Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Plot MSE for each model
plt.figure(figsize=(10, 5))  # Adjust the size as needed
for model_name, history in history_dict.items():
    plt.plot(history.history['mse'], label=f'{model_name} - Train MSE')
    plt.plot(history.history['val_mse'], label=f'{model_name} - Validation MSE')
plt.title('Training and Validation MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()  # Adjust the padding between and around subplots.
plt.show()
plt.show()

# Plot MAE for each model
plt.figure(figsize=(10, 5))  # Adjust the size as needed
for model_name, history in history_dict.items():
    plt.plot(history.history['mae'], label=f'{model_name} - Train MAE')
    plt.plot(history.history['val_mae'], label=f'{model_name} - Validation MAE')
plt.title('Training and Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
# # Set up the matplotlib figure with a grid of subplots
# fig, axes = plt.subplots(num_models, 3, figsize=(15, 5 * num_models), dpi=300)
#
# # Iterate over each model and its history, plotting the metrics
# for i, (model_name, history) in enumerate(history_dict.items()):
#     # Plot Loss
#     axes[i, 0].plot(history.history['loss'], label='Train Loss')
#     axes[i, 0].plot(history.history['val_loss'], label='Validation Loss')
#     axes[i, 0].set_title(f'{model_name} - Loss')
#     axes[i, 0].set_xlabel('Epoch')
#     axes[i, 0].set_ylabel('Loss')
#     axes[i, 0].legend()
#
#     # Plot MSE
#     axes[i, 1].plot(history.history['mse'], label='Train MSE')
#     axes[i, 1].plot(history.history['val_mse'], label='Validation MSE')
#     axes[i, 1].set_title(f'{model_name} - MSE')
#     axes[i, 1].set_xlabel('Epoch')
#     axes[i, 1].set_ylabel('MSE')
#     axes[i, 1].legend()
#
#     # Plot MAE
#     axes[i, 2].plot(history.history['mae'], label='Train MAE')
#     axes[i, 2].plot(history.history['val_mae'], label='Validation MAE')
#     axes[i, 2].set_title(f'{model_name} - MAE')
#     axes[i, 2].set_xlabel('Epoch')
#     axes[i, 2].set_ylabel('MAE')
#     axes[i, 2].legend()
#
# plt.tight_layout()
# plt.show()
# Bar Plot of Minimum Metrics Across All Models

# Initialize lists to store minimum metrics for each model
min_loss = []
min_mse = []
min_mae = []
model_names = list(history_dict.keys())

# Extract the minimum metrics from each model's history
for model_name, history in history_dict.items():
    min_loss.append(min(history.history['val_loss']))
    min_mse.append(min(history.history['val_mse']))
    min_mae.append(min(history.history['val_mae']))

# Create bar plots for each metric
x = range(num_models)

plt.figure(figsize=(15, 5), dpi=300)

# Minimum Loss
plt.subplot(1, 3, 1)
plt.bar(x, min_loss, color='blue')
plt.xticks(x, model_names, rotation='vertical')
plt.title('Minimum Validation Loss')
plt.ylabel('Loss')

# Minimum MSE
plt.subplot(1, 3, 2)
plt.bar(x, min_mse, color='orange')
plt.xticks(x, model_names, rotation='vertical')
plt.title('Minimum Validation MSE')
plt.ylabel('MSE')

# Minimum MAE
plt.subplot(1, 3, 3)
plt.bar(x, min_mae, color='green')
plt.xticks(x, model_names, rotation='vertical')
plt.title('Minimum Validation MAE')
plt.ylabel('MAE')

plt.tight_layout()
plt.show()

# Iterate through each model's history
data = []
for model_name, history in history_dict.items():
    for epoch in range(len(history.history['loss'])):
        # Create a dictionary for each epoch
        epoch_data = {
            'Model': model_name,
            'Epoch': epoch + 1,  # epochs are zero-indexed
            'Train Loss': history.history['loss'][epoch],
            'Validation Loss': history.history['val_loss'][epoch],
            'Train MSE': history.history['mse'][epoch],
            'Validation MSE': history.history['val_mse'][epoch],
            'Train MAE': history.history['mae'][epoch],
            'Validation MAE': history.history['val_mae'][epoch]
        }
        data.append(epoch_data)

# Create a DataFrame from the data
df_history = pd.DataFrame(data)

# Save or display the DataFrame as needed
print(df_history)

# Optionally, to save this table to a CSV file
df_history.to_csv("LSTMallModelEvaluationHistory.csv", index=False)