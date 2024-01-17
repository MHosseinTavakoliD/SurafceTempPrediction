import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['XGBoost', 'Transformer', 'Deep LSTM', 'Bidirectional LSTM', 'Stacked LSTM']
train_mae = [1.98025, 3.063226321231888, 5.620159149169922, 4.53136682510376, 5.520240783691406]
val_mae = [2.55392, 3.16882280080265, 5.817325592041016, 4.876107692718506, 6.01]
train_mse = [8.280178900900001, 21.93255511000256, 59.90994644165039, 42.0517463684082, 58.679832458496094]
val_mse = [15.622888656400002, 22.893868492310304, 65.19917297363281, 49.429656982421875, 64.26355743408203]

# Bar width
bar_width = 0.35

# Setting bar locations for each model
bar_locs = np.arange(len(models))

# Create the figure and the axes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bars for training MAE
train_mae_bars = ax1.bar(bar_locs - bar_width/2, train_mae, width=bar_width, label='Train MAE', color='orange', alpha=1)

# Bars for validation MAE
val_mae_bars = ax1.bar(bar_locs + bar_width/2, val_mae, width=bar_width, label='Validation MAE', color='red', alpha=1)

# Set labels and title for the primary y-axis
ax1.set_ylabel('Mean Absolute Error (MAE)', color='black')
ax1.set_xticks(bar_locs)
ax1.set_xticklabels(models, rotation=45, ha='right')
ax1.set_title('Model Training and Validation MAE and MSE')

# Instantiate a second y-axis sharing the same x-axis
ax2 = ax1.twinx()

# Bars for training MSE with transparency
train_mse_bars = ax2.bar(bar_locs - bar_width/2, train_mse, width=bar_width, label='Train MSE', color='green', alpha=1)

# Bars for validation MSE with transparency
val_mse_bars = ax2.bar(bar_locs + bar_width/2, val_mse, width=bar_width, label='Validation MSE', color='blue', alpha=1)

# Set labels for the secondary y-axis
ax2.set_ylabel('Mean Squared Error (MSE)', color='black')

# Add the legend by specifying the bars
ax1.legend(handles=[train_mae_bars, val_mae_bars, train_mse_bars, val_mse_bars],
           labels=['Train MAE', 'Validation MAE', 'Train MSE', 'Validation MSE'],
           loc='upper left', bbox_to_anchor=(0, 1), ncol=2, fancybox=True, shadow=True)
ax1.set_ylim(0, 7)
ax2.set_ylim(0, 85)
# Show the figure
plt.tight_layout()
plt.show()

# ****************************************************************************************************
import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['XGBoost', 'Transformer', 'Deep LSTM', 'Bidirectional LSTM', 'Stacked LSTM']
train_mae = [1.98025, 3.063226321231888, 5.620159149169922, 4.53136682510376, 5.520240783691406]
val_mae = [2.55392, 3.16882280080265, 5.817325592041016, 4.876107692718506, 6.01]
train_mse = [8.280178900900001, 21.93255511000256, 59.90994644165039, 42.0517463684082, 58.679832458496094]
val_mse = [15.622888656400002, 22.893868492310304, 65.19917297363281, 49.429656982421875, 64.26355743408203]

# Bar width
bar_width = 0.35

# Setting bar locations for each model
bar_locs = np.arange(len(models))

# Create the figure and the axes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Hatching patterns for black and white print
patterns = ['/', '\\', '|', '-', '+']

# Color mapping for each bar
colors = ['orange', 'red', 'green', 'blue']

# Bars for training MAE
train_mae_bars = ax1.bar(bar_locs - bar_width/2, train_mae, width=bar_width, label='Train MAE', color='#FFD580', hatch=patterns[0], edgecolor='black')

# Bars for validation MAE
val_mae_bars = ax1.bar(bar_locs + bar_width/2, val_mae, width=bar_width, label='Validation MAE', color='#FF9999', hatch=patterns[1], edgecolor='black')

# Set labels and title for the primary y-axis
ax1.set_ylabel('Mean Absolute Error (MAE)', color='black')
ax1.set_xticks(bar_locs)
ax1.set_xticklabels(models, rotation=45, ha='right')
ax1.set_title('Model Training and Validation MAE and MSE')

# Instantiate a second y-axis sharing the same x-axis
ax2 = ax1.twinx()

# Bars for training MSE with hatching
train_mse_bars = ax2.bar(bar_locs - bar_width/2, train_mse, width=bar_width, label='Train MSE', color='#90EE90', hatch=patterns[2], edgecolor='black')

# Bars for validation MSE with hatching
val_mse_bars = ax2.bar(bar_locs + bar_width/2, val_mse, width=bar_width, label='Validation MSE', color='#ADD8E6', hatch=patterns[3], edgecolor='black')

# Set labels for the secondary y-axis
ax2.set_ylabel('Mean Squared Error (MSE)', color='black')

# Add the legend by specifying the bars
legend = ax1.legend(handles=[train_mae_bars, val_mae_bars, train_mse_bars, val_mse_bars],
           labels=['Train MAE', 'Validation MAE', 'Train MSE', 'Validation MSE'],
           loc='upper left', bbox_to_anchor=(0, 1), ncol=2, fancybox=True, shadow=True)

# Adjust layout to make space for the legend
plt.subplots_adjust(right=0.75)  # Adjust the right edge of the subplot to make space for the legend
# Set y-axis limits
ax1.set_ylim(0, max(train_mae + val_mae) * 1.1)
ax2.set_ylim(0, max(train_mse + val_mse) * 1.1)

# Show the figure
plt.tight_layout()
plt.show()
