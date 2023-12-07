import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
forecast_horizon = 6
look_back = 24
Save_model = 'xgboost_model1Pred6h.pkl'
# Load and preprocess data
DataSource_file = 'C:/Users/zmx5fy/SurafceTempPrediction/Step4BuildingupDBforML/DBforMLaddingPredictionColAfterAfterCleaning/FinalDatasetForML6HourForecast.csv'
df = pd.read_csv(DataSource_file)
df['MeasureTime'] = pd.to_datetime(df['MeasureTime'])


# Add additional preprocessing if necessary

# Define your function to create the dataset for XGBoost
def create_dataset_for_xgboost(data, look_back=look_back, forecast_horizon=forecast_horizon):
    unique_stations = data['Station_name'].unique()
    X, Y = [], []

    for station in unique_stations:
        station_data = data[data['Station_name'] == station].copy()
        station_data.sort_values('MeasureTime', inplace=True)

        for i in range(len(station_data) - look_back - forecast_horizon + 1):
            past_features = station_data.iloc[i:i + look_back].drop(['MeasureTime', 'Station_name', 'County'], axis=1)
            target = station_data.iloc[i + look_back:i + look_back + forecast_horizon]['Surface TemperatureF']

            X.append(past_features.values.flatten())
            Y.append(target.values)

    return np.array(X), np.array(Y)


# Create dataset


X, Y = create_dataset_for_xgboost(df, look_back=look_back, forecast_horizon=forecast_horizon)

# Splitting the data
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define the XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror',
                         colsample_bytree=0.3,
                         learning_rate=0.1,
                         max_depth=5,
                         alpha=10,
                         n_estimators=200)

# Evaluation sets
eval_set = [(X_train, Y_train), (X_val, Y_val)]

# Train the model with evaluation metric
model.fit(X_train, Y_train, eval_metric="rmse", eval_set=eval_set, verbose=True, early_stopping_rounds=10)

# Save the model
# joblib.dump(model, Save_model)

# Visualization of Metrics
results = model.evals_result()
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='Validation')
ax.legend()
plt.ylabel('RMSE')
plt.title('XGBoost RMSE')
plt.show()

# Predict and Evaluate
predictions = model.predict(X_val)
val_mse = mean_squared_error(Y_val, predictions)
val_mae = mean_absolute_error(Y_val, predictions)
print(f'Mean Squared Error: {val_mse}')
print(f'Mean Absolute Error: {val_mae}')

# Visualization of Predicted vs Actual Values
# Assuming 'look_back' is 24 hours and 'forecast_horizon' is also 6 hours


list_index = [134, 462, 368, 529, 453, 375]  # Replace with indices of your choice
feature_index_for_surface_temp = 2  # Index for the surface temperature feature
number_of_features = X_train.shape[1] // look_back
print ("number_of_features", number_of_features)
for random_index in list_index:
    actual_data = Y_val[random_index].flatten()
    predicted_data = predictions[random_index].flatten()

    # Extract past surface temperature from X_val
    past_surface_temp = X_val[random_index].reshape(look_back, number_of_features)[:, feature_index_for_surface_temp]

    plt.figure(figsize=(15, 6))

    # Plot past surface temperature
    plt.plot(range(-look_back, 0), past_surface_temp, label='Past Surface Temp', color='green')

    # Plot actual surface temperature for the forecast horizon
    plt.plot(range(0, forecast_horizon), actual_data, label='Actual', color='blue')

    # Plot predicted surface temperature for the forecast horizon
    plt.plot(range(0, forecast_horizon), predicted_data, label='Predicted', color='red')

    plt.title(f'Surface Temperature Prediction for Station Index: {random_index}')
    plt.xlabel('Time Steps (Relative to Prediction Point)')
    plt.ylabel('Temperature')
    plt.legend()
    plt.show()
