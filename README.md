First step:
    Gathering data
    folder: Step1CollectingData
        FirstGetStationsFromWisconsinAPI.py
            Terget: preparing a list of stations from Wisconsin API
            This file gets the inforamtion (station ID) and name of each station (commented out).
            The list of stations is there to use in the SecondGet...py file
            I wrote some notes how each station is passing the data

        SecondGetDataHourlyFromWiscAPI.py
            Target: Scrape the Wisconsin API
            This code connects to Wisconsin API based on the station_id from the previous file, capture the data
            you can change the year, month, day, hour ...
            Right now, the data captured monthly and saves in the folder "records"

Second Step:
    folder: Step2DataCollection
        Here is basically, where we store the data from stations with good readings. So far, we just have the data from
        Wisconsin 2023 from Jan to Mid Nov. every 5 minutes

Third Step:
    folder: DataCleaning
    Target: The main target here is clear/standardize the data for ML purpose.
        WiscDataCleaning.py
            Standardizing the data based on these headings.
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
            The standard data is saved in the CleanDB folder as a CSV file. There is a little visualization/description in the CleanDB folder

Fourth Step:
    folder: Step4BuildingupDBforML
    Target: Modifying and clearing the data for using in the ML
        Clustering: location, data continuity, removing excess data (current data is every 10 min we want it every hour)
        file: DBforMLgenerationBeforeCleaning.py
            In this file we draw the hourly filterd data and stored the image in the folder: MonthlyGraphsHourlyBeforeCleaning
            In these graphs, we saw that some points are zero, so in a separate file I will do all again and also remove the zeros
        file: DBforMLgenerationAfterCleaning.py
            As mentioned, in this code, I removed the zeros and replace it with an average of a reading before and a reading after, if it is
            just one zero in between, if there are more than ten zero (ten hours), the data row is removed
        file: DBforMLaddingPredictionColAfterAfterCleaning.py
            I added columns regarding the predictions. In the LSTM modeling, I noticed that we need to add prediction columns to take them into account for predicting the
            surface temperature

Fifth Step:
    folder: Step5RNN_LSTM_model
    Target: Creating the Step5RNNLSTMmodel
    file: train1.py is the model based on the dataset without the prediction columns added
    file: train2.py is the models based on the dataset with prediction columns (e.g. we have columns of 1hr, .., 6hr forecast columns for predicting the suraface temp in the next 6 hours
    I've checked two structures of LSTM one is very simple, and the next one is more complicated

Sixth Step:
    Folder: Step6TransformerModel
    Target: Creating the Transformer model
    We tried 2 different packages: keras and pytorch
    Pytorch needed to have the same target and source size. which was kinda impossible or impractical for such database
    Keras was ok, but the output was the same for all the predictions. I tried some ways around, working on the hyperparamters, epoch range, and coding but
    the problem is not solved yet
    Update: Fixed pytorch got good results: first removed nan rows, then use a very simple model structure

Seventh Step:
    folder: Step7XGBosst
    Target: Creating the XGBosst model: done

Eighth Step:
    folder: Step8LightGBM
    Target: Creating the LightGBM model:
    This model can predict just one output, so for each hour forecast I trained a separate model.