import pandas as pd

from Step4BuildingupDBforML.DBforMLgenerationBeforeCleaning import plot_temperature_by_station_and_month

# Load your dataset
df = pd.read_csv('/Step4BuildingupDBforML/MonthlyGraphsHourlyBeforeCleaning/BeforeProcessedHourly_dataset.csv')

def modify_zeros(df, column):
    processed_dfs = []
    drop_indices_global = []
    for station in df['Station_name'].unique():
        print (station)
        df_station = df[df['Station_name'] == station].copy()
        # print (df_station)
        df_station.sort_values('MeasureTime', inplace=True)

        start_zero = None

        for index, row in df_station.iterrows():
            # print (f"index: {index} . row:  {row}")
            if row[column] == 0:
                start_zero = index if start_zero is None else start_zero
            else:
                if start_zero is not None:
                    zero_length = index - start_zero
                    if zero_length >= 10:
                        # df_station.drop(df_station.index[start_zero:index], inplace=True)
                        drop_indices_global.extend(range(start_zero, index))
                    else:
                        increment = (row[column] - df_station.loc[start_zero - 1, column]) / (zero_length + 1)
                        for i in range(start_zero, index):
                            df_station.loc[i, column] = df_station.loc[start_zero - 1, column] + (i - start_zero) * increment
                    start_zero = None
        # Handling end cases: if start_zero is not None at the end, it means the series of zeros continues till the end
        # Depending on requirement, either drop these rows or leave as is
                # Drop the accumulated rows

        processed_dfs.append(df_station)
    print (drop_indices_global)
    processed_dfs = pd.concat(processed_dfs)
    # processed_dfs = processed_dfs.drop(drop_indices_global)
    return processed_dfs, drop_indices_global



# df.to_csv('processed_datasetTestBefore.csv', index=False)

processed_df, Air_drop_list      = modify_zeros(df, 'Air TemperatureF')
print ("Done Air Temp cleaning")
processed_df,  Surface_drop_list = modify_zeros(processed_df, 'Surface TemperatureF')
print ("Done Surface Temp cleaning")
All_drop_list = list(set(Air_drop_list + Surface_drop_list))

# Drop the rows and save the processed DataFrame
processed_df = processed_df.drop(All_drop_list)

processed_df.to_csv('AfterProcessedHourly_dataset.csv', index=False)

df_draw = pd.read_csv('MonthlyGraphsHourlyAfterCleaning/AfterProcessedHourly_dataset.csv')
# Convert 'MeasureTime' to datetime format for filtering
df_draw['MeasureTime'] = pd.to_datetime(df_draw['MeasureTime'])


plot_temperature_by_station_and_month(df_draw, 'MonthlyGraphsHourlyAfterCleaning/')
