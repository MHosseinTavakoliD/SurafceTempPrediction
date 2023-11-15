import requests
import json
import pandas as pd
# from listOfSensors import Sensors_data
import openpyxl
from FirstGetStationsFromWisconsinAPI import Remaining_stations
from requests.exceptions import Timeout



Station_ID=["bbae3f19-5eef-4595-a583-271bd9cea23f"]
Station_name = "Newville"


# Request a new access token by logging in
request_URL = "https://www.viewmondousa.com/Token"
credentials = {
    "grant_type": "password",
    "Username": "MichiganTech",
    "Password": "Husky#1"
}

response = requests.post(request_URL, data=credentials)
print (response)

for station in Remaining_stations:
    print("HHHHHHHHHHHHHHHHHHHHHHHHHHH Station HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH ")
    Station_name = str(station["StationName"])
    Station_ID = str(station["StationId"])
    print (Station_name, Station_ID)
    #  Get sensor list
    try:
        if response.status_code == 200:
            if 'access_token' in response.text:
                access_token = response.json()["access_token"]  # Extract the new access token
                print("Access token:", access_token)
                # You can now use the new access token to make authorized requests to other endpoints.
                api_url = ("https://www.viewmondousa.com/api/v1/GetStationSensors?station_id=" + Station_ID )
                # "https://www.viewmondousa.com/api/v1/GetLastMeasureValues?Station_id=8c9541f8-8a52-4a2a-b2d6-ddf64b9bc764"
                headers = {
                    "Authorization": f"Bearer {access_token}"
                }

                Sensors_data = requests.get(api_url, headers=headers, timeout=200)
                # print (Sensors_data)
                Sensors_data = list(eval(Sensors_data.text))
                print('Sensor data', Sensors_data)
            else:
                print("Sensor data stage: Authorization failed. Check your credentials or the server response.")
        else:
            print(f"Request failed with status code {response.status_code}:")
            # print(response.text)

    except Timeout:
        pass

    ## write the date of each station in a txt file with the name of the staion


    for month in range(1,13):
        if month < 10: month = "0" + str(month)
        else: month = str(month)
        structured_data = []
        file_name = "./records/station." + Station_name + ".month" + month+".txt"
        with open(file_name, "w") as file:
            df = pd.DataFrame()

            for day in range(1, 2):

                if day < 10: day = "0" + str(day)
                else: day = str(day)
                for hour in range (0, 1):
                    if hour < 9 :
                        hour_from = "0" + str(hour)
                        hour_to = "0" + str(hour+1)
                    elif hour == 9:
                        hour_from = "0" + str(hour)
                        hour_to = "10"
                    else:
                      hour_from = str(hour)
                      hour_to = str(hour + 1)
                    print ("month: ", month ,"  day: ", day, "  hour: ", hour_from )
                    timeS = "from_time=2023-" + month + "-" + day + "T" + hour_from + "%3A00%3A00&to_time=2023-" + month + "-" + day + "T" + hour_to + "%3A00%3A00"


                    try:
                        if response.status_code == 200:
                            if 'access_token' in response.text:
                                access_token = response.json()["access_token"]  # Extract the new access token
                                print("Access token:", access_token)
                  # You can now use the new access token to make authorized requests to other endpoints.
                                api_url = ("https://www.viewmondousa.com/api/v1/GetMeasureValues?station_id="+ Station_ID+ "&" + timeS)
                                    #"https://www.viewmondousa.com/api/v1/GetLastMeasureValues?Station_id=8c9541f8-8a52-4a2a-b2d6-ddf64b9bc764"
                                headers = {
                                    "Authorization": f"Bearer {access_token}"
                                }
                                response1 = requests.get("https://www.viewmondousa.com/api/v1/")

                                response1 = requests.get(api_url, headers=headers, timeout=100)

                                print('Repsone each time', response1.text)
                            else:
                                print("Authorization failed. Check your credentials or the server response.")
                    except Timeout:
                        # response1 = "<Response [210]>"
                        pass
                    else:
                        print(f"Request failed with status code {response.status_code}:")
                        # print(response.text)

                    if response1.status_code != 200:
                        print ("Response 1 Error!!!!", response1.text)
                    if response1.status_code == 200:
                        print ("Response 1 NoError!!!!", response1.text)
                    else:
                        pass

                    try:
                        list_ = json.loads(response1.text)
                        i = 0

                        sensor_id_to_name = {sensor["SensorChannelId"]: sensor["SensorChannelName"] + sensor["SensorChannelUnit"] for sensor in Sensors_data}
                        print ("FFFFFFFFFFFFFFFFFFFFFFFFFF",sensor_id_to_name)
                        # Iterate through the data and create the desired structured representation


                        for item in list_:
                            measure_time = item["MeasureTime"]
                            measure_values = item["MeasureValues"]

                            # Create a dictionary for each time with SensorChannelName and data
                            time_data = {"MeasureTime": measure_time}
                            for value in measure_values:
                                sensor_channel_id = value["SensorChannelId"]
                                sensor_channel_name = sensor_id_to_name.get(sensor_channel_id, "Unknown Sensor")
                                sensor_data =   value["Value"]

                                time_data[sensor_channel_name] = sensor_data

                            structured_data.append(time_data)

                        # Now, you have a list of dictionaries with the desired structure

                        pd.set_option('display.max_columns', None)
                        pd.set_option('display.expand_frame_repr', False)

                        df = pd.DataFrame(structured_data)
                        print(df)




                        # for item in structured_data:
                        #     print (item)
                        # Write the result in Excel
                        # try:
                        #     with pd.ExcelWriter('output.xlsx', engine='openpyxl', mode='a') as writer:
                        #         df.to_excel(writer, index=False, sheet_name='Sheet1')
                        # except Exception as e:
                        #     print(f"An error occurred: {str(e)}")
                    except Exception as e:
                        # Code to handle the exception
                        print(f"An exception occurred: {e}")

            # Adjust formatting options

            formatted_df = df.to_string(index=False)
            # Write the DataFrame to a text file with formatting options
            file.write(formatted_df)