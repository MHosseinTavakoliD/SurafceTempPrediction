import os
import pandas as pd
import re
from DataCollection.Wisc2023.Locations import location_of_stations



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
your_directory_path = "C:/Users/zmx5fy/SurafceTempPrediction/DataCollection/Station_under_review//"



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
print (dataframes_list)

# Now dataframes_list contains all the DataFrames for each file, with added station info


