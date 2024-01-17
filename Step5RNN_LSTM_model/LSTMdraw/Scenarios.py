import matplotlib.pyplot as plt
# Detailed graph for Scenario 0
hours = range(48)
Scenarios = {
    "Scenario1":{
                    "past_surface_temp":  [89.959999, 86.539993, 87.979996, 87.440002, 88.339996, 89.239998, 85.279999, 85.279999, 85.279999, 86.720001, 88.160004, 88.160004, 88.699997, 88.880005, 89.059998, 88.160004, 88.339996, 89.059998,89.059998, 87.619995, 88.880005, 87.080002, 86.360001, 88.519997],
                    "actual_data" : [88.519997, 90.679993, 90.32, 87.979996, 91.039993, 93.559998, 94.099998, 94.82, 95.179993, 94.639999, 95., 95.899994, 96.079994, 97.339996, 98.599998, 99.679993, 100.939995, 99.679993, 97.339996, 93.379997, 89.599998, 85.279999, 84.199997, 85.279999, 88.880005],
                    "deep_lstm_pred" : [89.41598, 89.32927, 89.09375, 88.87766, 88.79526, 89.25544, 89.49941, 89.97004, 90.85585, 91.79951, 92.31428, 93.14778, 93.81582, 95.03014, 95.12618, 94.91715, 94.4411, 94.04386, 93.127, 92.349144,92.16049, 92.48445, 92.31938, 92.17773],
                    "bi_lstm_pred" : [88.400925, 88.70247, 89.21792, 89.75848, 90.58023, 90.61776, 90.00113, 88.992714, 88.76325, 88.914085,89.71521, 90.74594, 91.46461, 92.45041, 92.88302, 92.07102, 91.08354, 90.8263, 89.09368, 87.107994, 85.4473, 84.28028, 83.87027, 84.42893],
                    "stacked_lstm_pred" : [86.716064, 86.07484, 85.113266, 85.55894, 86.18511, 87.470055, 89.37265, 91.32523, 94.23943,96.32264, 98.53216, 99.01227, 99.72225, 99.59719, 98.47801, 96.5353, 94.91579, 93.37273,91.85227, 91.012856, 90.91735, 90.569115, 90.618355, 90.25959]

    },
    "Scenario2":{
                    "past_surface_temp":  [100.939995, 98.059998, 95.179993, 58.099998, 57.199997, 56.66, 55.580002, 59.18, 67.099998, 76.82, 85.639999, 92.120003, 97.159996, 99.679993, 100.939995, 96.260002, 90.139999, 82.759995, 75.739998, 72.32, 69.260002, 65.479996, 64.039993, 62.599998],
                    "actual_data" : [62.599998, 61.519997, 62.419998, 60.259998, 59., 57.739998, 57.919998, 60.080002, 69.080002, 77.899994, 85.82, 91.039993, 95.539993, 94.459999, 91.759995, 87.440002, 83.119995, 78.080002, 74.660004, 71.239998, 70.339996, 69.080002, 67.279999, 67.279999, 66.919998],
                    "deep_lstm_pred" : [65.77231, 63.10918, 60.956493, 59.56148, 59.87853, 61.97573, 65.7687, 71.81544, 77.94667, 84.13969, 89.21381, 92.074974, 92.79861, 92.16446, 89.84078, 86.35291, 82.842606, 79.256966, 76.2925, 73.60033, 71.63741, 69.79855, 67.63097, 65.54088],
                    "bi_lstm_pred" : [73.20809, 70.69019, 68.958855, 67.872826, 69.573296, 72.559654, 77.27837, 84.03577, 90.09791, 96.37113, 99.73698, 100.779465, 100.35306, 97.461716, 94.147964, 90.42557, 87.47545, 83.107834, 79.50329, 77.14871, 75.46739, 74.19034, 71.739, 70.31175],
                    "stacked_lstm_pred" : [63.8979, 61.74368, 59.762398, 58.28266, 57.72236, 59.28737, 63.2201, 69.62448, 77.031815, 84.36628, 90.85365, 94.5482, 95.97727, 95.33875, 92.11194, 87.469, 81.79076, 76.11355, 70.44882, 65.85057, 63.11217, 60.67061, 58.860558, 58.0692]

    },
    "Scenario3":{
                    "past_surface_temp":   [44.059998, 41.540001, 36.860001, 36.860001, 44.599998, 54.860001, 58.82, 66.380005, 69.979996, 73.760002, 68.720001, 63.139999, 54.860001, 49.099998, 45.68, 43.16, 41.18, 38.48, 37.579998, 36.68, 36.32, 40.279999, 37.220001, 38.48],
                    "actual_data" : [38.48, 40.82, 35.779999, 40.459999, 44.419998, 52.699997, 59., 61.16, 62.959999, 60.98, 56.66, 54.139999, 51.98, 45.68, 42.440002, 40.279999, 39.02, 37.040001, 74.479996, 73.220001, 71.599998, 70.339996, 69.800003, 68.539993, 68.360001],
                    "deep_lstm_pred" : [37.089363, 39.37825, 42.346893, 46.951794, 52.116043, 57.324516, 61.86188, 63.781815, 63.656246, 61.17095, 57.593143, 53.491627, 50.076057, 48.987415, 48.005733, 49.463257, 52.918255, 56.638943, 59.467922, 63.29247, 66.185745, 68.869026, 69.09295, 68.474556],
                    "bi_lstm_pred" : [38.12725, 40.185986, 44.866165, 50.708233, 57.851627, 64.11096, 68.67357, 69.464676, 67.657455, 62.950424, 56.882145, 50.39464, 45.625816, 41.32704, 38.43265, 37.26567, 36.724056, 36.13562, 35.74633, 36.868073, 37.07848, 37.18153, 38.00502, 40.129345],
                    "stacked_lstm_pred" :  [42.66298, 43.31096, 44.77088, 47.71205, 49.75771, 51.4196, 53.617588, 54.76696, 54.29677, 53.957897, 52.7287, 51.806366, 50.55552, 50.56521, 51.18225, 52.482796, 54.555916, 56.385185, 57.141365, 58.472088, 59.301426, 59.918118, 60.182495, 60.99579]
    },
    "Scenario4":{
                        "past_surface_temp":   [48.019997, 41.900002, 34.52, 26.779999, 23.18, 22.82, 19.58, 18.860001, 23.540001, 25.34, 26.060001, 27.139999, 26.779999, 21.92, 21.559999, 23.900002, 28.940001, 37.040001, 45.139999, 50.720001, 53.599998, 53.959999, 50.900002, 48.559998],
                        "actual_data" : [48.559998, 44.059998, 41.18, 37.939999, 36.68, 35.779999, 30.559999, 33.98, 33.439999, 34.16, 33.98, 34.52, 34.34, 34.52, 34.52, 34.880001, 36.5, 37.220001, 37.759998, 38.66, 39.380001, 39.919998, 39.740002, 41.900002, 41.18],
                        "deep_lstm_pred" : [47.172646, 44.438198, 41.547104, 38.752525, 36.27137, 34.749237, 34.127377, 34.576675, 34.893345, 35.675182, 35.816944, 35.969738, 36.10555, 35.952217, 36.066334, 36.994286, 38.3329, 40.027885, 41.58435, 42.714893, 43.64573, 43.577118, 43.211895, 41.88169],
                        "bi_lstm_pred" :[47.107418, 43.72448, 40.057434, 36.512753, 33.887924, 32.583275, 32.080807, 32.25829, 31.95763, 32.52771, 33.388783, 33.387554, 34.10618, 34.312263, 36.10509, 36.742565, 37.95096, 38.951717, 39.35797, 38.88474, 38.67081, 37.7418, 35.641533, 33.7862],
                        "stacked_lstm_pred" :[41.19877, 38.72394, 36.10594, 34.412827, 33.264637, 32.799572, 32.666225, 32.928852, 32.3827, 32.189354, 31.685673, 31.215963, 30.661358, 31.578705, 32.78445, 34.12093, 36.19998, 38.50415, 40.56104, 41.602993, 42.092, 41.21144, 39.56047, 38.22901]
    },
    "Scenario5":{
                        "past_surface_temp":[70.699997, 73.760002, 79.339996, 82.940002, 93.379997, 99.32, 96.079994, 96.620003, 97.520004, 95.720001, 88.339996, 83.479996, 80.599998, 78.080002, 74.839996, 73.940002, 73.220001, 72.5, 71.419998, 70.699997, 69.080002, 67.82, 67.099998, 64.940002],
                        "actual_data" :[64.940002, 64.760002, 67.279999, 68.720001, 68.899994, 69.260002, 70.339996, 70.339996, 70.699997, 70.160004, 69.619995, 69.080002, 67.82, 68.360001, 68.539993, 67.279999, 65.479996, 68.360001, 68.720001, 68.539993, 68.18, 68.539993, 69.979996, 71.419998, 73.580002],
                        "deep_lstm_pred" :[73.130875, 72.34279, 72.4215, 72.47386, 71.78177, 71.11452, 70.22126, 69.23554, 68.46213, 67.565765, 67.09651, 66.97402, 67.06908, 66.89142, 66.19399, 66.41856, 66.71676, 67.55177, 68.9061, 70.47815, 71.55903, 72.776886, 72.92579, 73.42892],
                        "bi_lstm_pred" :[71.42227, 72.28648, 73.331116, 73.00736, 73.35961, 72.62839, 71.80591, 71.26925, 70.9433, 70.753044, 70.6353, 69.95137, 69.05103, 68.33075, 66.61558, 64.97153, 63.47207, 63.294975, 63.07565, 64.23906, 65.88813, 68.10156, 69.59287, 71.63475],
                        "stacked_lstm_pred" :[67.61735, 67.65, 68.27124, 68.674385, 69.584915, 69.83844, 70.22891, 70.27283, 70.25367, 69.67853, 69.074326, 68.87605, 68.10944, 67.22494, 66.94792, 66.47153, 66.146484, 66.21144, 66.92231, 66.89971, 67.07639, 67.80685, 68.10708, 68.94625]
    },
    "Scenario6":{
                        "past_surface_temp": [33.259998, 32.18, 31.639999, 31.280001, 30.92, 30.92, 31.280001, 31.280001, 30.02, 30.380001, 30.380001, 30.380001, 30.380001, 30.559999, 30.74, 31.459999, 32.18, 32.18, 32.900002, 32.900002, 33.259998, 32.720001, 32.720001, 32.],
                        "actual_data" :[32, 31.280001, 30.02, 29.299999, 9.140001, 9.860001, 8.960001, 9.680002, 10.400002, 10.040001, 17.060001, 24.08, 31.1, 39.200001, 47.119999, 50.900002, 50.720001, 47.119999, 42.619999, 34.52, 28.940001, 25.16, 23.360001, 22.639999, 22.279999],
                        "deep_lstm_pred" :[23.059595, 21.38552, 18.9954, 16.920938, 14.545235, 12.208405, 11.492566, 12.797846, 16.15553, 20.916798, 27.727615, 35.341312, 41.398956, 46.128643, 48.864876, 48.067772, 45.56619, 40.913628, 35.19185, 30.602816, 26.367693, 23.020832, 20.578121, 18.760944],
                        "bi_lstm_pred" :[28.284122, 26.90379, 26.030926, 24.172493, 22.712008, 20.521744, 18.805794, 17.608536, 18.346792, 20.977045, 25.381577, 31.836657, 37.521225, 41.79169, 45.121063, 44.923344, 43.23545, 39.301384, 35.364334, 30.612123, 26.766592, 23.103262, 21.072231, 20.331972],
                        "stacked_lstm_pred" :[23.381123, 22.708063, 22.128656, 20.183847, 18.672358, 16.561832, 14.590303, 14.628895, 16.013103, 20.13677, 25.33444, 31.27135, 36.967735, 42.04747, 46.08097, 46.692818, 45.088615, 42.11641, 38.113434, 32.991993, 28.240341, 24.93044, 22.714361, 21.589413]
},
}

title_font_size = 14
axis_label_font_size = 12
legend_font_size = 11
tick_label_font_size = 12

fig, axs = plt.subplots(2, 3, figsize=(15, 10), dpi=400)
fig.suptitle('Temperature Predictions across Scenarios', fontsize=16)
j = 0
for i, (scenario, data) in enumerate(Scenarios.items()):
    ax = axs[i // 3, i % 3]
    print (scenario, len(data['actual_data']))

    # Plotting data
    ax.plot(hours[:24], data['past_surface_temp'], label='Past Surface Temp', color='blue', linestyle='solid')
    ax.plot(hours[23:], data['actual_data'], label='Actual Data', color='green', linestyle='--')
    ax.plot(hours[24:], data['deep_lstm_pred'], label='Deep LSTM Predicted', color='red', linestyle='--', marker = '|', markersize=10,  markevery=5)
    ax.plot(hours[24:], data['bi_lstm_pred'], label='Bidirectional LSTM Predicted', color='purple', linestyle='--', marker = '*', markersize=7,  markevery=5)
    ax.plot(hours[24:], data['stacked_lstm_pred'], label='Stacked LSTM Predicted', color='orange', linestyle='--', marker = 'o', markersize=7,  markevery=5)

    # Adding titles and labels
    ax.set_title(f'{scenario}')
    ax.set_xlabel('Hours',fontsize=axis_label_font_size)
    ax.set_ylabel('Temperature', fontsize=axis_label_font_size)
    if j ==0:
        ax.legend(fontsize=legend_font_size)
    j = 1
    ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
