import os
import pandas as pd
import re
from DataCollection.Wisc2023.Locations import location_of_stations
import numpy as np


# Function to parse a single line using regular expressions to handle multiple spaces
def parse_line(line):
    return re.split(r'\s{2,}', line)

# Function to convert a text file into a DataFrame
def convert_txt_to_df(file_path, station_info):
    parsed_lines = []
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                parsed_lines.append(parse_line(line.strip()))
    df = pd.DataFrame(parsed_lines[1:], columns=parsed_lines[0])
    # Adding station information
    df['Station_name'] = station_info['Station_name']
    df['County'] = station_info['County']
    df['Latitude'] = station_info['Location'][0]
    df['Longitude'] = station_info['Location'][1]
    return df

# Function to extract station name from file name
def extract_station_name(filename):
    # Assuming the file name format is 'station.[station_name].month[--].txt'
    return filename.split('.')[1]

# Station_under_review
# your_directory_path = "C:/Users/zmx5fy/SurafceTempPrediction/DataCollection/Station_under_review//"
your_directory_path = "C:/Users/zmx5fy/SurafceTempPrediction/DataCollection/Wisc2023/"


# Mapping station names to their information
station_info_map = {info['Station_name']: info for info in location_of_stations}
dataframes_list = []
# Loop over each file in the directory
for filename in os.listdir(your_directory_path):
    # List to hold all DataFrames

    if filename.endswith('.txt'):  # Check if the file is a .txt file
        station_name = extract_station_name(filename)
        print ("*****************************",filename)
        station_info = station_info_map.get(station_name, None)
        if station_info:
            file_path = os.path.join(your_directory_path, filename)
            df = convert_txt_to_df(file_path, station_info)
            dataframes_list.append(df)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
# print (dataframes_list)

def standardize_dataframe(df, common_columns):
    # Ensure all required columns are present, fill missing ones with NaN
    for col in common_columns:
        if col not in df.columns:
            df[col] = np.nan

    # Reorder and select the columns to match the standard format
    standardized_df = df[common_columns]

    return standardized_df


common_columns = [
    'MeasureTime',
    'Rel. Humidity%',
    'Air Temperature°F',
    'Surface Temperature°F',
    'Wind Speed (act)mph',
    'Precipitation Intensityin/h',
    'Saline Concentration%',
    'Friction',
    'Ice Percent%',
    'Station_name',
    'County',
    'Latitude',
    'Longitude'
]

# Apply standardization to each DataFrame in the list
standardized_dfs = [standardize_dataframe(df, common_columns) for df in dataframes_list]

# Optionally, you can concatenate all standardized DataFrames into a single DataFrame
combined_standardized_df = pd.concat(standardized_dfs, ignore_index=True)

# Print the combined standardized DataFrame
print(combined_standardized_df)

# DBV1Wis23 = First Version of Database consists of 2023 from Jan to mid Nov Station are: Appleton, Arcadia, Kenosha, Manitowish, Neenah, Tipler
DB_file_name = './CleanDB/DBV1Wis23.csv'
with open(DB_file_name, "w") as file:
    formatted_df = combined_standardized_df.to_csv(index=False, encoding='utf-8')
    # Write the DataFrame to a text file with formatting options
    file.write(formatted_df)