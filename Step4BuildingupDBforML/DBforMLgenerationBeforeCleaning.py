import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# Visualisation: This function takes a DataFrame as input, loops through each station, then each month within each station,
# and generates the temperature plots. The DataFrame should have columns MeasureTime, Station_name, Air Temperature°F, and Surface Temperature°F.
out_dir = 'MonthlyGraphsHourlyBeforeCleaning/'

def plot_temperature_by_station_and_month(df, output_directory):
    """
    Plots and saves air and surface temperature vs time for each month and station.

    Parameters:
    df (pandas.DataFrame): DataFrame containing temperature data.
    output_directory (str): Directory to save the plots.
    """


    # Check if output directory exists, if not, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Loop through each station
    for station in df['Station_name'].unique():
        df_station = df[df['Station_name'] == station].copy()
        # Ensure data is sorted by time (important for plotting)
        df_station.sort_values('MeasureTime', inplace=True)
        # Loop through each month
        for month in df_station['MeasureTime'].dt.month.unique():
            df_month = df_station[df_station['MeasureTime'].dt.month == month].copy()

            # Plotting
            plt.figure(figsize=(20, 6))  # Increase the width here
            plt.plot(df_month['MeasureTime'], df_month['Air TemperatureF'], label='Air TemperatureF', alpha=0.5)
            plt.plot(df_month['MeasureTime'], df_month['Surface TemperatureF'], label='Surface TemperatureF', alpha=0.5)

            # Set fixed y-axis range
            plt.ylim(-40, 120)

            plt.title(f'Air and Surface Temperature for {station} - Month: {month} (Hourly Data)')
            plt.xlabel('Time')
            plt.ylabel('Temperature (°F)')
            plt.legend()
            plt.grid(True)

            # Formatting the date on the x-axis for clarity
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())  # Use AutoDateLocator for better tick management
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))  # Set interval to 1 to show every day
            plt.gcf().autofmt_xdate()  # Rotate date labels

            # Save the plot with a wider format
            plot_filename = f"{station}_Month{month}_HourlyData_wide.png"
            plt.savefig(os.path.join(output_directory, plot_filename))
            plt.close()  # Close the plot to prevent it from displaying

# Assuming 'DBV1Wis23.csv' is your csv file with the hourly data already filtered
df = pd.read_csv('C:/Users/zmx5fy/SurafceTempPrediction/DataCleaning/CleanDB/DBV1Wis23.csv', encoding='ISO-8859-1')

# Convert 'MeasureTime' to datetime format for filtering
df['MeasureTime'] = pd.to_datetime(df['MeasureTime'])

# Filtering for data within the first ten minutes of each hour
df_hourly = df[(df['MeasureTime'].dt.minute < 10)]

# Now df_hourly contains only the data for each hour
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# Calling the function with the hourly data
plot_temperature_by_station_and_month(df_hourly, out_dir)
