import pandas as pd

# Load and preprocess data
df = pd.read_csv('C:/Users/zmx5fy/SurafceTempPrediction/Step4BuildingupDBforML/MonthlyGraphsHourlyAfterCleaning/AfterProcessedHourly_dataset.csv')

stations = df['Station_name'].unique()

# Initialize an empty DataFrame to hold the new dataset
new_dataset = pd.DataFrame()

for station in stations:
    # Filter data for the current station
    station_data = df[df['Station_name'] == station]

    # Sort data by time to ensure proper forecasting
    station_data = station_data.sort_values(by='MeasureTime')

    # Forecast columns for air temperature, humidity, wind velocity, and precipitation
    for hour in range(1, 7):
        station_data[f'AirTempForecast_{hour}hr'] = station_data['Air TemperatureF'].shift(-hour)
        station_data[f'HumidityForecast_{hour}hr'] = station_data['Rel. Humidity%'].shift(-hour)
        station_data[f'WindVelocityForecast_{hour}hr'] = station_data['Wind Speed (act)mph'].shift(-hour)
        station_data[f'PrecipitationForecast_{hour}hr'] = station_data['Precipitation Intensityin/h'].shift(-hour)

    # Append the processed data for this station to the new dataset
    new_dataset = pd.concat([new_dataset, station_data], ignore_index=True)


# Saving the new dataset to a CSV file
new_dataset.to_csv('C:/Users/zmx5fy/SurafceTempPrediction/Step4BuildingupDBforML/DBforMLaddingPredictionColAfterAfterCleaning/'
                   'FinalDatasetForML6HourForecast.csv', index=False)

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
# df = pd.read_csv(
#     '/Step4BuildingupDBforML/DBforMLaddingPredictionColAfterAfterCleaning/FinalDatasetForML12HourForecast.csv', parse_dates=['MeasureTime'])
#
#
#
# # Specify the time period (e.g., '2023-01-01' to '2023-01-07')
# start_date = '2023-01-06'
# end_date = '2023-01-07'
#
# # Create the mask for the specified time period
# mask = (df['MeasureTime'] >= start_date) & (df['MeasureTime'] <= end_date)
#
# # Apply the mask to filter the data
# filtered_data = df.loc[mask]
#
# # Assuming you are focusing on a single station, filter by station name (e.g., 'Appleton')
# station_data = filtered_data[filtered_data['Station_name'] == 'Appleton']
#
# # Plotting
# plt.figure(figsize=(15, 6))
# plt.plot(station_data['MeasureTime'], station_data['Air TemperatureF'], label='Actual Air Temperature')
# print ("station_data['Air TemperatureF']",station_data['Air TemperatureF'])
# # Plot each of the 24-hour forecast data
# TM = [1]
# for i in TM:
#     plt.plot(station_data['MeasureTime'], station_data[f'AirTempForecast_{i}hr'], label=f'{i}hr Forecast')
#     print ("station_data[f'AirTempForecast_{i}hr", station_data[f'AirTempForecast_{i}hr'])
#
# plt.xlabel('Time')
# plt.ylabel('Air Temperature (F)')
# plt.title('Air Temperature and 24-hour Forecasts at Appleton Station for Specified Period')
# plt.legend()
# plt.show()
#
