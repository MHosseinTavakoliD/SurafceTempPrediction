import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

out_dir = 'MonthlyGraphsHourlyBeforeCleaning/'

def plot_temperature_for_kenosha(df, output_directory):
    # Filter for the Kenosha station
    df_kenosha = df[df['Station_name'] == 'Kenosha'].copy()

    # Ensure data is sorted by time (important for plotting)
    df_kenosha.sort_values('MeasureTime', inplace=True)
    print(df_kenosha)
    # Check if output directory exists, if not, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Loop through each month
    for month in df_kenosha['MeasureTime'].dt.month.unique():
        print (month)
        df_month = df_kenosha[df_kenosha['MeasureTime'].dt.month == month].copy()

        # Plotting
        plt.figure(figsize=(20, 6))
        plt.plot(df_month['MeasureTime'], df_month['Air TemperatureF'], label='Air TemperatureF', alpha=0.5)
        plt.plot(df_month['MeasureTime'], df_month['Surface TemperatureF'], label='Surface TemperatureF', alpha=0.5)

        # Set fixed y-axis range
        plt.ylim(-40, 120)

        plt.title(f'Air and Surface Temperature for Kenosha - Month: {month} (Hourly Data)')
        plt.xlabel('Time')
        plt.ylabel('Temperature (°F)')
        plt.legend()
        plt.grid(True)

        # Formatting the date on the x-axis
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()  # Rotate date labels

        # Save the plot with a wider format and fixed y-axis range
        plot_filename = f"Kenosha_Month{month}_HourlyData.png"
        plt.savefig(os.path.join(output_directory, plot_filename))
        plt.close()  # Close the plot

# Assuming 'DBV1Wis23.csv' is your csv file with the hourly data already filtered
df = pd.read_csv('C:/Users/zmx5fy/SurafceTempPrediction/DataCleaning/CleanDB/DBV1Wis23.csv', encoding='ISO-8859-1')

# Convert 'MeasureTime' to datetime format for filtering
df['MeasureTime'] = pd.to_datetime(df['MeasureTime'])

# Filtering for hourly data
df_hourly = df[df['MeasureTime'].dt.minute == 0]

print (df_hourly)
# Set display options for pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# Call the function with the hourly data for Kenosha
plot_temperature_for_kenosha(df_hourly, out_dir)
