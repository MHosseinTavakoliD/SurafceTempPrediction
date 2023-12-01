import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, Input, GlobalAveragePooling1D
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Parameters
Epoch = 1000
LR = 0.005
forecast_horizon = 6
look_back = 24
file_model_save = 'TransformerSurfaceTempForecastModel6H.h5'
DataSource_file = 'C:/Users/zmx5fy/SurafceTempPrediction/Step4BuildingupDBforML/DBforMLaddingPredictionColAfterAfterCleaning/FinalDatasetForML6HourForecast.csv'

# Load and preprocess data
df = pd.read_csv(DataSource_file)
df['MeasureTime'] = pd.to_datetime(df['MeasureTime'])

# Function to create dataset
def create_dataset(data, look_back=24, forecast_horizon=12):
    unique_stations = data['Station_name'].unique()
    X, Y, stations = [], [], []
    for station in unique_stations:
        print (station)
        station_data = data[data['Station_name'] == station].copy()
        station_data.sort_values('MeasureTime', inplace=True)
        for i in range(len(station_data) - look_back - forecast_horizon + 1):
            current_time = station_data.iloc[i + look_back - 1]['MeasureTime']
            future_time = station_data.iloc[i + look_back]['MeasureTime']
            if (future_time - current_time).total_seconds() > 3600:
                continue
            past_features = station_data.iloc[i:i + look_back].drop(['MeasureTime', 'Station_name', 'County'], axis=1)
            target = station_data.iloc[i + look_back:i + look_back + forecast_horizon]['Surface TemperatureF']
            X.append(past_features)
            Y.append(target.values)
            stations.append(station)
    return np.array(X), np.array(Y), np.array(stations)

# Create dataset
X, Y, station_names = create_dataset(df, look_back=look_back, forecast_horizon=forecast_horizon)

# Split data
train_X, val_X, train_Y, val_Y = {}, {}, {}, {}
for station in np.unique(station_names):
    idx = station_names == station
    X_train, X_val, Y_train, Y_val = train_test_split(X[idx], Y[idx], test_size=0.2, random_state=42)
    train_X[station] = X_train
    val_X[station] = X_val
    train_Y[station] = Y_train
    val_Y[station] = Y_val
print (X_val.shape)
# Transformer Block
def TransformerBlock(embed_dim, num_heads, ff_dim, rate=0.1):
    inputs = Input(shape=(None, embed_dim))
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    attention_output = Dropout(rate)(attention_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    ffn_output = Dense(ff_dim, activation="relu")(out1)
    ffn_output = Dense(embed_dim)(ffn_output)
    ffn_output = Dropout(rate)(ffn_output)
    return Model(inputs=inputs, outputs=LayerNormalization(epsilon=1e-6)(out1 + ffn_output))

def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0.1, mlp_dropout=0.1):
    inputs = Input(shape=input_shape)
    x = Dense(head_size)(inputs)  # Ensuring the embedding dimension matches
    for _ in range(num_transformer_blocks):
        x = TransformerBlock(head_size, num_heads, ff_dim, dropout)(x)
    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(forecast_horizon)(x)
    return Model(inputs=inputs, outputs=outputs)

# Update model parameters if necessary
embed_dim = 34  # Ensure this matches the last dimension of your input data
num_heads = 2
ff_dim = 136
num_transformer_blocks = 4
mlp_units = [128]

# Combine all stations into one dataset
combined_train_X = np.concatenate(list(train_X.values()))
combined_train_Y = np.concatenate(list(train_Y.values()))
combined_val_X = np.concatenate(list(val_X.values()))
combined_val_Y = np.concatenate(list(val_Y.values()))

# Train the model on the combined dataset
model = build_transformer_model(
    input_shape=combined_train_X.shape[1:],
    head_size=embed_dim,
    num_heads=num_heads,
    ff_dim=ff_dim,
    num_transformer_blocks=num_transformer_blocks,
    mlp_units=mlp_units,
    mlp_dropout=0.4,
    dropout=0.25,
)
model.compile(optimizer=Adam(LR), loss='mean_squared_error', metrics=['mse', 'mae'])
history = model.fit(combined_train_X, combined_train_Y, epochs=Epoch, batch_size=32, validation_data=(combined_val_X, combined_val_Y), verbose=1)

# Save the model
model.save(file_model_save)

# Visualization of Training
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 3, 2)
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('Model MSE')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 3, 3)
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

# Combine all validation data into one array
combined_val_X = np.concatenate(list(val_X.values()))

# Predict using the combined validation data
predictions = model.predict(combined_val_X)

# Combine all validation target data into one array
combined_val_Y = np.concatenate(list(val_Y.values()))

# Calculate MSE and MAE
val_mse = mean_squared_error(combined_val_Y, predictions)
val_mae = mean_absolute_error(combined_val_Y, predictions)
print(f'Validation MSE: {val_mse}, Validation MAE: {val_mae}')

# Visualization of Predictions
# ... (You can use your existing code for visualization here)
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

    print(predictions)
    # Display the first few predictions
    print("First few predictions:", predictions[:5])

    # Check if all values are the same
    if np.all(predictions == predictions[0]):
        print("Warning: All predicted values are the same:", predictions[0])
    else:
        print("Predicted values vary, which is expected behavior.")
