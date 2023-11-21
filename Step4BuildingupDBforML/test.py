import pandas as pd
from DBforMLgenerationBeforeCleaning import plot_temperature_by_station_and_month

# Load your dataset
df = pd.read_csv('/Step3DataCleaning/CleanDB/test.csv')


def modify_zeros(df, column):
    processed_dfs = []
    for station in df['Station_name'].unique():
        df_station = df[df['Station_name'] == station].copy()
        df_station.sort_values('MeasureTime', inplace=True)
        print (df_station)

        # Initialize variables to track the start of zero values sequence
        start_zero = None
        for index, row in df_station.iterrows():
            if row[column] == 0:
                start_zero = index if start_zero is None else start_zero
                print ("here zero starts " , start_zero)
            else:
                if start_zero is not None:
                    # Check the length of consecutive zeros
                    if index - start_zero >= 10:
                        df_station.drop(df_station.index[start_zero:index], inplace=True)
                        print ("Should delete here ", index)
                    else:
                        # Calculate the average increment
                        increment = (row[column] - df_station.loc[start_zero - 1, column]) / (index - start_zero + 1)
                        for i in range(start_zero, index):
                            df_station.loc[i, column] = df_station.loc[start_zero - 1, column] + (
                                        i - start_zero + 1) * increment
                        print ("For less than 10 increment is ", increment)
                start_zero = None

        processed_dfs.append(df_station)

    # Combine all processed stations data
    return pd.concat(processed_dfs)


# Apply the function to your dataframe and column of interest
processed_df = modify_zeros(df, 'Air TemperatureF')
processed_df = modify_zeros(processed_df, 'Surface TemperatureF')  # Apply again for Surface Temperature

print (processed_df)
# # Save the processed DataFrame
# processed_df.to_csv('processed_dataset.csv', index=False)
#
# # Convert 'MeasureTime' to datetime format for filtering
# processed_df['MeasureTime'] = pd.to_datetime(processed_df['MeasureTime'])
# plot_temperature_by_station_and_month(processed_df, 'MonthlyGraphsHourlyAfterCleaning/')