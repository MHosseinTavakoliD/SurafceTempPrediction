import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
# Recom: make the bias false, have to rebuild the transformer from the scratch....
# self.transformer = nn.Transformer instead use self.transformer = Transformer.encoder

# Load and preprocess data
df = pd.read_csv('C:/Users/zmx5fy/SurafceTempPrediction/Step4BuildingupDBforML/DBforMLaddingPredictionColAfterAfterCleaning/FinalDatasetForML6HourForecast.csv')  # Update this path
df['MeasureTime'] = pd.to_datetime(df['MeasureTime'])
forecast_horizon=6
look_back=24
Learning_Rate = 0.001
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
print ("X", X.shape)
print ("Y", Y.shape)

# Splitting the data
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
# print ("X_val", X_val)
# print ("X_val[0]", X_val[0])
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
X_train_tensor = X_train_tensor[~nan_mask_train]
Y_train_tensor = Y_train_tensor[~nan_mask_train]

# Step 2: Identify and remove rows with NaN in validation data
nan_mask_val = torch.isnan(X_val_tensor).any(dim=2).any(dim=1)
X_val_tensor = X_val_tensor[~nan_mask_val]
Y_val_tensor = Y_val_tensor[~nan_mask_val]

# Check the new shapes of the tensors
print("Shape of X_train_tensor after NaN removal:", X_train_tensor.shape)
print("Shape of Y_train_tensor after NaN removal:", Y_train_tensor.shape)
print("Shape of X_val_tensor after NaN removal:", X_val_tensor.shape)
print("Shape of Y_val_tensor after NaN removal:", Y_val_tensor.shape)


# Define a Transformer Model
class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_features, num_layers, num_heads, dim_feedforward, dropout=0.1, num_output_features=forecast_horizon):
        super(TimeSeriesTransformer, self).__init__()
        self.transformer = nn.Transformer(
            d_model=num_features,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.output_layer = nn.Linear(num_features, num_output_features)  # Using output_layer for clarity

    def forward(self, src):
        # Assuming src is of shape [batch_size, sequence_length, num_features]
        memory = self.transformer.encoder(src)
        output = self.output_layer(memory[:, -1, :])  # Taking the last time step
        return output


# Initialize the model
num_features = X_train_tensor.shape[2]  # Number of features
print ("num_features",num_features)
model = TimeSeriesTransformer(num_features, num_layers=1, num_heads=2, dim_feedforward=2048, dropout=0.1, num_output_features=forecast_horizon)

# Training and evaluation functions
def train_epoch(model, train_loader, optimizer, criterion, clip_value=1.0):
    # print ("Call: train_epoch")
    model.train()
    total_loss = 0
    for src, tgt in train_loader:
        optimizer.zero_grad()
        output = model(src)
        # Ensure tgt is reshaped or sliced to match output shape
        loss = criterion(output, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)  # Gradient clipping

        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion):
    # print("Call: evaluate")
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in val_loader:
            output = model(src)
            # Ensure tgt is reshaped or sliced to match output shape
            loss = criterion(output, tgt)
            total_loss += loss.item()
    return total_loss / len(val_loader)

# Training loop
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
# print ("val_dataset[0]", val_dataset[0])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
optimizer = torch.optim.Adam(model.parameters(), lr=Learning_Rate)
criterion = nn.MSELoss()
# Define Learning Rate Scheduler
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

train_losses, val_losses = [], []
for epoch in range(Epoch):
    for param_group in optimizer.param_groups:
        print(f'Current learning rate: {param_group["lr"]}')
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss = evaluate(model, val_loader, criterion)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f'Epoch {epoch}: Train Loss: {train_loss}, Val Loss: {val_loss}')

    # Step the scheduler
    # scheduler.step()

# Save model
torch.save(model.state_dict(), 'model_state_dict.pt')
# torch.save(model, 'model.pt')

# Visualization of training and validation loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses[1:], label='Train Loss')
plt.plot(val_losses[1:], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()

# Prediction
predictions = []
model.eval()
# i =0
with torch.no_grad():
    for X in val_loader:
        i =+ 1
        # print ("X[0][0]", X[0][0])
        output = model(X[0])  # X[0] is the source tensor
        # print ("output", output)
        predictions.extend(output.cpu().numpy())
        # if i < 3:
        #     print("X[0].shape", X[0].shape)
        #     print ("predictions", predictions)

predictions = np.array(predictions).squeeze()
surface_temp_index = 2

# Visualization of Predictions with past observations
for random_index in [134, 462, 368, 529, 453, 375]:
    past_data = X_val_tensor[random_index, :, surface_temp_index].numpy()
    actual_data = Y_val_tensor[random_index].numpy()  # Assuming Y_val_tensor is [num_samples, forecast_horizon]
    predicted_data = predictions[random_index]

    print (random_index)
    print ("past_data", past_data.shape, past_data)
    print ("actual_data", actual_data.shape,actual_data)
    print ("predicted_data", predicted_data.shape, predicted_data)

    plt.figure(figsize=(15, 6))
    plt.plot(range(-look_back, 0), past_data, label='Past', color='green')
    plt.plot(range(0, forecast_horizon), actual_data, label='Actual', color='blue')
    plt.plot(range(0, forecast_horizon), predicted_data, label='Predicted', color='red')
    plt.title(f'Surface Temperature Prediction for Station Index: {random_index}')
    plt.xlabel('Time Steps')
    plt.ylabel('Temperature')
    plt.legend()
    plt.show()