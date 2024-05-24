import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt

# Constants
look_back = 24  # Historical hours to use as input
forecast_horizon = 24  # Future hours to predict
n_estimators = 250

# Load and preprocess data
data_source_file = 'modified_DatasetForSmartMDSS.csv'
df = pd.read_csv(data_source_file)
df['MeasureTime'] = pd.to_datetime(df['MeasureTime'])


# Extract time components and encode cyclical features
def encode_cyclical_feature(df, col, max_vals):
    df[col + '_sin'] = np.sin(2 * np.pi * df[col] / max_vals)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col] / max_vals)
    return df


df['hour'] = df['MeasureTime'].dt.hour
df['day_of_week'] = df['MeasureTime'].dt.dayofweek
df['day_of_month'] = df['MeasureTime'].dt.day
df['month'] = df['MeasureTime'].dt.month
df['year'] = df['MeasureTime'].dt.year
df = encode_cyclical_feature(df, 'hour', 24)
df = encode_cyclical_feature(df, 'day_of_week', 7)
df = encode_cyclical_feature(df, 'month', 12)
print("Number of columns in the DataFrame:", df.shape[1])
print("Number of columns in the DataFrame:", len(df.columns))


# Drop 'MeasureTime' and 'Station_name'
df = df.drop(['MeasureTime', 'Station_name'], axis=1)


# Define function to create dataset for XGBoost
def create_dataset_for_xgboost(data, look_back, forecast_horizon):
    X, Y = [], []
    total_rows = len(data)

    for i in tqdm(range(total_rows - look_back - forecast_horizon + 1)):
        # Extract historical features for X
        past_features = data.iloc[i:i + look_back]
        # Extract forecast features for X, excluding 'Surface TemperatureF'
        forecast_features = data.iloc[i + look_back:i + look_back + forecast_horizon].drop(['Surface TemperatureF'],
                                                                                           axis=1)

        # Combine past and forecast features to form the complete X
        X_features = pd.concat([past_features, forecast_features]).values.flatten()
        X.append(X_features)

        # Extract 'Surface TemperatureF' from forecast data for Y
        Y_target = data.iloc[i + look_back:i + look_back + forecast_horizon]['Surface TemperatureF'].values
        Y.append(Y_target)

    return np.array(X), np.array(Y)


# Create dataset
X, Y = create_dataset_for_xgboost(df, look_back, forecast_horizon)

# Splitting the data
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define the XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror',
                         colsample_bytree=0.5,
                         learning_rate=0.1,
                         max_depth=5,
                         alpha=10,
                         n_estimators=n_estimators)

# Evaluation sets
eval_set = [(X_train, Y_train), (X_val, Y_val)]

# Train the model with evaluation metrics
model.fit(X_train, Y_train, eval_metric=["rmse", "mae"], eval_set=eval_set, verbose=True)

# Save the model
joblib.dump(model, 'xgboost_model.pkl')

# Plotting training progress
results = model.evals_result()
num_boost_rounds = len(results['validation_0']['rmse'])
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

boosting_rounds = range(1, num_boost_rounds + 1)
ax1.plot(boosting_rounds, results['validation_0']['rmse'], label='Training RMSE')
ax1.plot(boosting_rounds, results['validation_1']['rmse'], label='Validation RMSE')
ax1.legend()
ax1.set_xlabel('Boosting Rounds')
ax1.set_ylabel('RMSE Value')
ax1.set_title('XGBoost RMSE')

ax2.plot(boosting_rounds, results['validation_0']['mae'], label='Training MAE')
ax2.plot(boosting_rounds, results['validation_1']['mae'], label='Validation MAE')
ax2.legend()
ax2.set_xlabel('Boosting Rounds')
ax2.set_ylabel('MAE Value')
ax2.set_title('XGBoost MAE')

plt.tight_layout()
plt.show()
