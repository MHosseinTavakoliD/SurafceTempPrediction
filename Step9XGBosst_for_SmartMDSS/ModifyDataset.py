import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('FinalDatasetForSmartMdss.csv')

# List of columns to remove
columns_to_remove = [
        "Saline Concentration%","Friction","Ice Percent%","County",
    'AirTempForecast_1hr', 'HumidityForecast_1hr', 'WindVelocityForecast_1hr', 'PrecipitationForecast_1hr',
    'AirTempForecast_2hr', 'HumidityForecast_2hr', 'WindVelocityForecast_2hr', 'PrecipitationForecast_2hr',
    'AirTempForecast_3hr', 'HumidityForecast_3hr', 'WindVelocityForecast_3hr', 'PrecipitationForecast_3hr',
    'AirTempForecast_4hr', 'HumidityForecast_4hr', 'WindVelocityForecast_4hr', 'PrecipitationForecast_4hr',
    'AirTempForecast_5hr', 'HumidityForecast_5hr', 'WindVelocityForecast_5hr', 'PrecipitationForecast_5hr',
    'AirTempForecast_6hr', 'HumidityForecast_6hr', 'WindVelocityForecast_6hr', 'PrecipitationForecast_6hr',
    'AirTempForecast_7hr', 'HumidityForecast_7hr', 'WindVelocityForecast_7hr', 'PrecipitationForecast_7hr',
    'AirTempForecast_8hr', 'HumidityForecast_8hr', 'WindVelocityForecast_8hr', 'PrecipitationForecast_8hr',
    'AirTempForecast_9hr', 'HumidityForecast_9hr', 'WindVelocityForecast_9hr', 'PrecipitationForecast_9hr',
    'AirTempForecast_10hr', 'HumidityForecast_10hr', 'WindVelocityForecast_10hr', 'PrecipitationForecast_10hr',
    'AirTempForecast_11hr', 'HumidityForecast_11hr', 'WindVelocityForecast_11hr', 'PrecipitationForecast_11hr',
    'AirTempForecast_12hr', 'HumidityForecast_12hr', 'WindVelocityForecast_12hr', 'PrecipitationForecast_12hr',
    'AirTempForecast_13hr', 'HumidityForecast_13hr', 'WindVelocityForecast_13hr', 'PrecipitationForecast_13hr',
    'AirTempForecast_14hr', 'HumidityForecast_14hr', 'WindVelocityForecast_14hr', 'PrecipitationForecast_14hr',
    'AirTempForecast_15hr', 'HumidityForecast_15hr', 'WindVelocityForecast_15hr', 'PrecipitationForecast_15hr',
    'AirTempForecast_16hr', 'HumidityForecast_16hr', 'WindVelocityForecast_16hr', 'PrecipitationForecast_16hr',
    'AirTempForecast_17hr', 'HumidityForecast_17hr', 'WindVelocityForecast_17hr', 'PrecipitationForecast_17hr',
    'AirTempForecast_18hr', 'HumidityForecast_18hr', 'WindVelocityForecast_18hr', 'PrecipitationForecast_18hr',
    'AirTempForecast_19hr', 'HumidityForecast_19hr', 'WindVelocityForecast_19hr', 'PrecipitationForecast_19hr',
    'AirTempForecast_20hr', 'HumidityForecast_20hr', 'WindVelocityForecast_20hr', 'PrecipitationForecast_20hr',
    'AirTempForecast_21hr', 'HumidityForecast_21hr', 'WindVelocityForecast_21hr', 'PrecipitationForecast_21hr',
    'AirTempForecast_22hr', 'HumidityForecast_22hr', 'WindVelocityForecast_22hr', 'PrecipitationForecast_22hr',
    'AirTempForecast_23hr', 'HumidityForecast_23hr', 'WindVelocityForecast_23hr', 'PrecipitationForecast_23hr',
    'AirTempForecast_24hr', 'HumidityForecast_24hr', 'WindVelocityForecast_24hr', 'PrecipitationForecast_24hr'
]

# Remove the specified columns
df = df.drop(columns=columns_to_remove)

# Save the modified DataFrame back to a CSV file
df.to_csv('modified_DatasetForSmartMDSS.csv', index=False)
