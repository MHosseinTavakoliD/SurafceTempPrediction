# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, Input, GlobalAveragePooling1D
from keras.optimizers import Adam
from kerastuner import HyperModel, RandomSearch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

# Parameters
Epoch = 20
forecast_horizon = 6
look_back = 24
file_model_save = 'TransformerSurfaceTempForecastModel6H.keras'
DataSource_file = 'C:/Users/zmx5fy/SurafceTempPrediction/Step4BuildingupDBforML/DBforMLaddingPredictionColAfterAfterCleaning/FinalDatasetForML6HourForecast.csv'

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
def create_dataset(data, look_back=24, forecast_horizon=6):
    unique_stations = data['Station_name'].unique()
    X, Y, stations = [], [], []
    for station in unique_stations:
        print (station)
        station_data = data[data['Station_name'] == station].copy()
        station_data.sort_values('MeasureTime', inplace=True)
        loop_range = len(station_data) - look_back - forecast_horizon + 1
        for i in tqdm(range(loop_range), desc=f"Processing Data for {station}", leave=False):
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
# Combine all stations into one dataset
combined_train_X = np.concatenate(list(train_X.values()))
combined_train_Y = np.concatenate(list(train_Y.values()))
combined_val_X = np.concatenate(list(val_X.values()))
combined_val_Y = np.concatenate(list(val_Y.values()))
# Define the Transformer Block as a function (same as your existing code)
def TransformerBlock(embed_dim, num_heads, ff_dim, rate=0.001):
    inputs = Input(shape=(None, embed_dim))
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    attention_output = Dropout(rate)(attention_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    ffn_output = Dense(ff_dim, activation="relu")(out1)
    ffn_output = Dense(embed_dim)(ffn_output)
    ffn_output = Dropout(rate)(ffn_output)
    return Model(inputs=inputs, outputs=LayerNormalization(epsilon=1e-6)(out1 + ffn_output))
# Define the Hypermodel class for Keras Tuner
class TransformerHyperModel(HyperModel):
    def __init__(self, input_shape, forecast_horizon):
        self.input_shape = input_shape
        self.forecast_horizon = forecast_horizon

    def build(self, hp):
        # Hyperparameters to tune
        lr = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
        num_heads = hp.Int('num_heads', min_value=2, max_value=10, step=2)
        ff_dim = hp.Int('ff_dim', min_value=64, max_value=256, step=64)
        num_transformer_blocks = hp.Int('num_transformer_blocks', min_value=1, max_value=5, step=1)
        mlp_units = hp.Int('mlp_units', min_value=64, max_value=256, step=64)
        dropout_rate = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
        mlp_dropout = hp.Float('mlp_dropout', min_value=0.1, max_value=0.5, step=0.1)

        # Model construction (similar to your existing code but with hyperparameters)
        inputs = Input(shape=self.input_shape)
        x = Dense(num_heads * 2)(inputs)  # Adjust the embedding dimension
        for _ in range(num_transformer_blocks):
            x = TransformerBlock(num_heads * 2, num_heads, ff_dim, dropout_rate)(x)
        x = GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in [mlp_units]:
            x = Dense(dim, activation="relu")(x)
            x = Dropout(mlp_dropout)(x)
        outputs = Dense(self.forecast_horizon)(x)
        model = Model(inputs=inputs, outputs=outputs)

        # Compile model
        model.compile(optimizer=Adam(lr), loss='mean_squared_error', metrics=['mse', 'mae'])
        return model

# Combine all stations into one dataset (as per your existing code)

# Create an instance of your hypermodel
hypermodel = TransformerHyperModel(input_shape=combined_train_X.shape[1:], forecast_horizon=forecast_horizon)

# Use Random Search for hyperparameter tuning
tuner = RandomSearch(
    hypermodel,
    objective='val_mse',
    max_trials=10,
    executions_per_trial=2,
    directory='my_dir',
    project_name='transformer_tuning'
)

# Start tuning
tuner.search(combined_train_X, combined_train_Y, epochs=10, validation_data=(combined_val_X, combined_val_Y))

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of heads is {best_hps.get('num_heads')},
feed-forward network dimension is {best_hps.get('ff_dim')}, number of transformer blocks is {best_hps.get('num_transformer_blocks')},
learning rate is {best_hps.get('learning_rate')}, MLP units are {best_hps.get('mlp_units')},
dropout rate is {best_hps.get('dropout')}, and MLP dropout rate is {best_hps.get('mlp_dropout')}.
""")

# You can then build the final model with these optimal hyperparameters and train it on your data
