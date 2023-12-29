import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV
data = pd.read_csv('result.csv')

# Create a figure and a set of subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

# Plotting Loss values
ax1.plot(data['Epoch'], data['Train Loss'], label='Train Loss')
ax1.plot(data['Epoch'], data['Val Loss'], label='Val Loss')
ax1.set_title('Training and Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend(loc='upper right')

# Plotting MAE values
ax2.plot(data['Epoch'], data['Train MAE'], label='Train MAE')
ax2.plot(data['Epoch'], data['Val MAE'], label='Val MAE')
ax2.set_title('Training and Validation MAE')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MAE')
ax2.legend(loc='upper right')

# Adding Learning Rate as a secondary y-axis to the first graph
ax1b = ax1.twinx()
ax1b.plot(data['Epoch'], data['Learning Rate'], label='Learning Rate', color='grey', linestyle='dashed')
ax1b.set_ylabel('Learning Rate')
ax1b.legend(loc='upper left')

# Adjusting layout
plt.tight_layout()

# Show plot
plt.show()
