import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load and preprocess data
df = pd.read_csv('C:/Users/zmx5fy/SurafceTempPrediction/Step4BuildingupDBforML/DBforMLaddingPredictionColAfterAfterCleaning/FinalDatasetForML6HourForecast.csv')  # Update this path
df['MeasureTime'] = pd.to_datetime(df['MeasureTime'])

# Function to create dataset
def create_dataset(data, look_back=6, forecast_horizon=6):
    unique_stations = data['Station_name'].unique()
    X, Y = [], []
    surface_temp_index = data.columns.get_loc("Surface TemperatureF")  # Replace with your actual column name
    for station in unique_stations:
        station_data = data[data['Station_name'] == station].copy()
        station_data.sort_values('MeasureTime', inplace=True)
        loop_range = len(station_data) - look_back - forecast_horizon + 1
        for i in tqdm(range(loop_range), desc=f"Processing Data for {station}", leave=False):
            current_time = station_data.iloc[i + look_back - 1]['MeasureTime']
            future_time = station_data.iloc[i + look_back]['MeasureTime']
            if (future_time - current_time).total_seconds() > 3600:
                continue
            past_features = station_data.iloc[i:i + look_back].drop(['MeasureTime', 'Station_name', 'County'], axis=1)
            target = station_data.iloc[i + look_back:i + look_back + forecast_horizon, surface_temp_index]
            X.append(past_features.values)
            Y.append(target.values)
    return np.array(X), np.array(Y)

# Create dataset
X, Y = create_dataset(df)

# Splitting the data
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)

print ("X_train_tensor", X_train_tensor.shape)
print("X_val_tensor", X_val_tensor.shape)
print("Y_train_tensor", Y_train_tensor.shape)
print("Y_val_tensor", Y_val_tensor.shape)

# Define a Transformer Model
class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_features, num_layers, num_heads, dim_feedforward, dropout=0.1, num_output_features=1):
        super(TimeSeriesTransformer, self).__init__()
        self.transformer = nn.Transformer(
            d_model=num_features,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.linear = nn.Linear(num_features, num_output_features)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        output = self.transformer(src, tgt, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask)
        return self.linear(output[-1])  # Assuming you want the final step's output for each input sequence

# Initialize the model
num_features = X_train_tensor.shape[2]  # Number of features
model = TimeSeriesTransformer(num_features, num_layers=3, num_heads=2, dim_feedforward=2048, dropout=0.1, num_output_features=1)

# Training and evaluation functions
def train_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for src, tgt in train_loader:
        optimizer.zero_grad()
        # Assuming the last dimension of tgt is 1
        output = model(src, tgt[:, :-1])  # Do not include the last time step of tgt which is to be predicted
        loss = criterion(output, tgt[:, -1])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in val_loader:
            output = model(src, tgt[:, :-1])  # Do not include the last time step of tgt
            loss = criterion(output, tgt[:, -1])
            total_loss += loss.item()
    return total_loss / len(val_loader)

# Training loop
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

train_losses, val_losses = [], []
for epoch in range(10):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss = evaluate(model, val_loader, criterion)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f'Epoch {epoch}: Train Loss: {train_loss}, Val Loss: {val_loss}')

# Save model
torch.save(model.state_dict(), 'model_state_dict.pt')
torch.save(model, 'model.pt')

# Visualization of training and validation loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()

# Prediction
predictions = []
model.eval()
with torch.no_grad():
    for X, _ in val_loader:
        output = model(X, torch.zeros_like(X))  # You might need to adjust this line if your model is set up differently
        predictions.extend(output.cpu().numpy())
predictions = np.array(predictions).squeeze()
surface_temp_index = 2
look_back = 6
forecast_horizon = 6
# Visualization of Predictions with past observations
for random_index in [134, 462, 368, 529, 453, 375]:
    past_data = X_val_tensor[random_index, :, surface_temp_index].numpy()
    actual_data = Y_val_tensor[random_index, :, 0].numpy()
    predicted_data = predictions[random_index]

    plt.figure(figsize=(15, 6))
    plt.plot(range(-look_back, 0), past_data, label='Past', color='green')
    plt.plot(range(0, forecast_horizon), actual_data, label='Actual', color='blue')
    plt.plot(range(0, forecast_horizon), predicted_data, label='Predicted', color='red')
    plt.title(f'Surface Temperature Prediction for Station Index: {random_index}')
    plt.xlabel('Time Steps')
    plt.ylabel('Temperature')
    plt.legend()
    plt.show()
