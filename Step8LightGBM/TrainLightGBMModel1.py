import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
import os

# Configuration
forecast_horizon = 24
look_back = 24
model_save_dir = './ModelForEachForecastHorizon/'
DataSource_file = 'C:/Users/zmx5fy/SurafceTempPrediction/Step4BuildingupDBforML/DBforMLaddingPredictionColAfterAfterCleaning/FinalDatasetForML24HourForecast.csv'
# Lists to store metrics for each model
mse_values = []
mae_values = []
# Ensure the directory for model saving exists
os.makedirs(model_save_dir, exist_ok=True)

# Load and preprocess data
df = pd.read_csv(DataSource_file)
df['MeasureTime'] = pd.to_datetime(df['MeasureTime'])

# Function to create the dataset for LightGBM for a specific target hour
def create_dataset_for_lightgbm(data, look_back=look_back, forecast_horizon=forecast_horizon):
    unique_stations = data['Station_name'].unique()
    X, Y = [], []

    for station in unique_stations:
        print (station)
        station_data = data[data['Station_name'] == station].copy()
        station_data.sort_values('MeasureTime', inplace=True)

        for i in range(len(station_data) - look_back - forecast_horizon + 1):
            past_features = station_data.iloc[i:i + look_back].drop(['MeasureTime', 'Station_name', 'County'], axis=1)
            target = station_data.iloc[i + look_back:i + look_back + forecast_horizon]['Surface TemperatureF']

            X.append(past_features.values.flatten())
            Y.append(target.values)

    return np.array(X), np.array(Y).reshape(-1, forecast_horizon)

# Create dataset
X, Y = create_dataset_for_lightgbm(df, look_back=look_back, forecast_horizon=forecast_horizon)

# Train and evaluate a model for each hour in the forecast horizon
for hour in range(forecast_horizon):
    Y_hour = Y[:, hour]  # Slice the target array for the specific hour
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y_hour, test_size=0.2, random_state=42)


    # Define the LightGBM model
    model = lgb.LGBMRegressor(objective='regression',
                              num_leaves=80,
                              learning_rate=0.1,
                              n_estimators=1000)

    # Train the model
    early_stopping_callback = lgb.early_stopping(stopping_rounds=10)
    model.fit(X_train, Y_train, eval_set=[(X_val, Y_val)], eval_metric='l1', callbacks=[early_stopping_callback])

    # Save the model
    model_save_path = os.path.join(model_save_dir, f'lightgbm_model_hour_{hour}.pkl')
    try:
        joblib.dump(model, model_save_path)
        print(f"Model for hour {hour} saved successfully at {model_save_path}")
    except Exception as e:
        print(f"Error saving model for hour {hour}: {e}")
    # Evaluation
    predictions = model.predict(X_val)
    val_mse = mean_squared_error(Y_val, predictions)
    val_mae = mean_absolute_error(Y_val, predictions)
    mse_values.append(val_mse)
    mae_values.append(val_mae)
    print(f'Hour: {hour}, Mean Squared Error: {val_mse}, Mean Absolute Error: {val_mae}')


# Plotting the MSE and MAE for each model
hours = [f'Hour {i}' for i in range(forecast_horizon)]

plt.figure(figsize=(15, 6))

# MSE Curve
plt.plot(hours, mse_values, label='Mean Squared Error', marker='o', color='red')

# MAE Curve
plt.plot(hours, mae_values, label='Mean Absolute Error', marker='o', color='blue')

plt.title('MSE and MAE for Each Hour-Model')
plt.xlabel('Model (Hour)')
plt.ylabel('Error Value')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Function to load a saved model
def load_model(hour):
    model_path = os.path.join(model_save_dir, f'lightgbm_model_hour_{hour}.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"No model found for hour {hour} at {model_path}")

# Visualization for each selected window
list_index = [134, 462, 368, 529, 453, 375]
number_of_features = X_train.shape[1] // look_back
feature_index_for_surface_temp = 2  # Adjust this index based on your data
# Create a 2D array to store actual future temperatures for each index in list_index
actual_future_temps = np.array([Y[index, :] for index in list_index])

for idx, index in enumerate(list_index):
    plt.figure(figsize=(15, 6))

    # Prepare the past surface temperature data for plotting
    past_surface_temp = X_val[index].reshape(look_back, number_of_features)[:, feature_index_for_surface_temp]

    # Plot past surface temperature
    plt.plot(range(-look_back, 0), past_surface_temp, label='Past Surface Temp', color='green')

    # Predict using each model and collect the predictions
    predictions = []
    for hour in range(forecast_horizon):
        model = load_model(hour)
        prediction = model.predict(X_val[index].reshape(1, -1))
        predictions.append(prediction[0])

    # Plot predicted surface temperature for the forecast horizon
    plt.plot(range(0, forecast_horizon), predictions, label='Predicted', color='red')

    # Plot actual surface temperature for the forecast horizon
    actual_future_temp = actual_future_temps[idx, :]  # Use the pre-stored actual temperatures
    plt.plot(range(0, forecast_horizon), actual_future_temp, label='Actual', color='blue')

    plt.title(f'Surface Temperature Prediction for Window Index: {index}')
    plt.xlabel('Time Steps (Relative to Prediction Point)')
    plt.ylabel('Temperature')
    plt.legend()
    plt.show()
