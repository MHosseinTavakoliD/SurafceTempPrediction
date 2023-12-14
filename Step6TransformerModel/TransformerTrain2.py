import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load the data
df = pd.read_csv('C:/Users/zmx5fy/SurafceTempPrediction/Step4BuildingupDBforML/MonthlyGraphsHourlyAfterCleaning/AfterProcessedHourly_dataset.csv', parse_dates=['MeasureTime'])

# Sort the data
df.sort_values(['Station_name', 'MeasureTime'], inplace=True)

# Check for missing data
print(df.isnull().sum())

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

# # Drop original datetime and non-numerical columns
# df.drop(['MeasureTime', 'Station_name', 'County', 'Latitude', 'Longitude'], axis=1, inplace=True)
#
# # Normalize the data
# scaler = MinMaxScaler()
# df_scaled = scaler.fit_transform(df)

# Windowing function

def create_windows(data, input_width, label_width, shift, target_column='Surface TemperatureF'):
    X = []
    y = []
    unique_stations = data['Station_name'].unique()

    for station in unique_stations:
        station_data = data[data['Station_name'] == station].copy()
        station_data.sort_values('MeasureTime', inplace=True)
        station_data.drop(['MeasureTime', 'Station_name', 'County'], axis=1, inplace=True)

        # Normalize the data for the station
        scaler = MinMaxScaler()
        station_data_scaled = scaler.fit_transform(station_data)

        loop_range = len(station_data) - input_width - label_width + 1
        for i in tqdm(range(loop_range), desc=f"Processing Data for {station}", leave=False):
            X.append(station_data_scaled[i:i + input_width])

            # Extract only the target column (surface temperature) for the next 12 hours
            y_start_index = i + input_width
            y_end_index = y_start_index + label_width
            y.append(station_data_scaled[y_start_index:y_end_index, station_data.columns.get_loc(target_column)])

    return np.array(X), np.array(y)


input_width = 24
label_width = 12
shift = 12

X, y = create_windows(df, input_width, label_width, shift)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Printing the shape of the training and validation data
print(f"X-Training data shape: {X_train.shape}")
print(f"X-Validation data shape: {X_val.shape}")
print(f"Y-Training data shape: {y_train.shape}")
print(f"Y-Validation data shape: {y_val.shape}")


