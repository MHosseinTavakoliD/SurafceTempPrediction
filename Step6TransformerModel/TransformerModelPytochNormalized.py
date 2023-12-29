import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load and preprocess data
df = pd.read_csv(
    'C:/Users/zmx5fy/SurafceTempPrediction/Step4BuildingupDBforML/DBforMLaddingPredictionColAfterAfterCleaning/FinalDatasetForML24HourForecast.csv')  # Update this path
df['MeasureTime'] = pd.to_datetime(df['MeasureTime'])
forecast_horizon = 24
look_back = 24
Learning_Rate = 0.0005
Epoch = 50
batch_size = 32

# Function to create dataset
def create_dataset(data, look_back=look_back, forecast_horizon=forecast_horizon):
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
print("X", X.shape)
print("Y", Y.shape)

# Splitting the data
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)

# print ("X_val_tensor[0]", X_val_tensor[0])


print ("X_train_tensor", X_train_tensor.shape)
print("X_val_tensor", X_val_tensor.shape)
print("Y_train_tensor", Y_train_tensor.shape)
print("Y_val_tensor", Y_val_tensor.shape)


# Step 1: Identify and remove rows with NaN in training data
nan_mask_train = torch.isnan(X_train_tensor).any(dim=2).any(dim=1)
X_train = X_train_tensor[~nan_mask_train]
Y_train = Y_train_tensor[~nan_mask_train]

# Step 2: Identify and remove rows with NaN in validation data
nan_mask_val = torch.isnan(X_val_tensor).any(dim=2).any(dim=1)
X_val = X_val_tensor[~nan_mask_val]
Y_val = Y_val_tensor[~nan_mask_val]

# Check the new shapes of the tensors
print("Shape of X_train_tensor after NaN removal:", X_train_tensor.shape)
print("Shape of Y_train_tensor after NaN removal:", Y_train_tensor.shape)
print("Shape of X_val_tensor after NaN removal:", X_val_tensor.shape)
print("Shape of Y_val_tensor after NaN removal:", Y_val_tensor.shape)

# Normalize the data
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

# Reshape the training data to make it two-dimensional
num_features = X_train.shape[2]  # Number of features
X_train_reshaped = X_train.reshape(-1, num_features)

X_train_normalized = scaler_X.fit_transform(X_train_reshaped)
X_val_normalized = scaler_X.transform(X_val.reshape(-1, num_features))

Y_train_normalized = scaler_Y.fit_transform(Y_train)
Y_val_normalized = scaler_Y.transform(Y_val)

# Reshape the normalized data back to its original shape
X_train_normalized = X_train_normalized.reshape(-1, look_back, num_features)
X_val_normalized = X_val_normalized.reshape(-1, look_back, num_features)

# Convert to PyTorch tensors
X_train_tensor_normalized = torch.tensor(X_train_normalized, dtype=torch.float32)
X_val_tensor_normalized = torch.tensor(X_val_normalized, dtype=torch.float32)
Y_train_tensor_normalized = torch.tensor(Y_train_normalized, dtype=torch.float32)
Y_val_tensor_normalized = torch.tensor(Y_val_normalized, dtype=torch.float32)

# Define a Transformer Model
class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_features, num_layers, num_heads, dim_feedforward, dropout=0.1,
                 num_output_features=forecast_horizon, bias=False):
        super(TimeSeriesTransformer, self).__init__()
        self.transformer = nn.Transformer(
            d_model=num_features,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            bias=bias  # Set bias to False to turn off biases
        )
        self.output_layer = nn.Linear(num_features, num_output_features)  # Using output_layer for clarity

    def forward(self, src):
        # Assuming src is of shape [batch_size, sequence_length, num_features]
        memory = self.transformer.encoder(src)
        output = self.output_layer(memory[:, -1, :])  # Taking the last time step
        return output

# Initialize the model
model = TimeSeriesTransformer(num_features, num_layers=1, num_heads=2, dim_feedforward=2048, dropout=0.1,
                              num_output_features=forecast_horizon, bias=False)

# Training and evaluation functions
def train_epoch(model, train_loader, optimizer, criterion, clip_value=1.0):
    model.train()
    total_loss = 0
    for src, tgt in train_loader:
        optimizer.zero_grad()
        output = model(src)
        loss = criterion(output, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)  # Gradient clipping
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in val_loader:
            output = model(src)
            loss = criterion(output, tgt)
            total_loss += loss.item()
    return total_loss / len(val_loader)

# Training loop
train_dataset = TensorDataset(X_train_tensor_normalized, Y_train_tensor_normalized)
val_dataset = TensorDataset(X_val_tensor_normalized, Y_val_tensor_normalized)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=Learning_Rate)
criterion = nn.MSELoss()

train_losses, val_losses = [], []
for epoch in range(Epoch):
    for param_group in optimizer.param_groups:
        print(f'Current learning rate: {param_group["lr"]}')
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss = evaluate(model, val_loader, criterion)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f'Epoch {epoch}: Train Loss: {train_loss}, Val Loss: {val_loss}')

# Save model
torch.save(model.state_dict(), 'TransformerPytorchBiasFalse.pt')

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
    for X in val_loader:
        output = model(X[0])  # X[0] is the source tensor
        predictions.extend(output.cpu().numpy())

predictions = np.array(predictions).squeeze()

# Inverse scale the predicted values to the original scale
predicted_data = scaler_Y.inverse_transform(predictions)
surface_temp_index = 2

# Visualization of Predictions with past observations
for random_index in [134, 462, 368, 529, 453, 375]:
    past_data = X_val_tensor_normalized[random_index, :, surface_temp_index].numpy()
    actual_data = scaler_Y.inverse_transform(Y_val_tensor_normalized[random_index].numpy())
    predicted_data = predicted_data[random_index]

    plt.figure(figsize=(15, 6))
    plt.plot(range(-look_back, 0), past_data, label='Past', color='green')
    plt.plot(range(0, forecast_horizon), actual_data, label='Actual', color='blue')
    plt.plot(range(0, forecast_horizon), predicted_data, label='Predicted', color='red')
    plt.title(f'Surface Temperature Prediction for Station Index: {random_index}')
    plt.xlabel('Time Steps')
    plt.ylabel('Temperature')
    plt.legend()
    plt.show()
