import matplotlib.pyplot as plt
import pandas as pd

# Load the data
data = pd.read_csv('resualt.csv')  # Make sure to put the correct path here

# Set global parameters for font size and colors, adjust as needed
plt.rcParams.update({
    'font.size': 12,  # Adjust font size here
    'axes.labelcolor': 'black',  # Change label colors
    'xtick.color': 'black',  # Change xtick color
    'ytick.color': 'black'  # Change ytick color
})

# Create subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

# Plotting MSE with specified colors and font sizes and adding major gridlines
axes[0].plot(data['Estimator'], data['Train_MSE'], label='Train MSE', color='blue')  # Change line color here
axes[0].plot(data['Estimator'], data['Val_MSE'], label='Validation MSE', color='green', linestyle='--')  # Change line color here
axes[0].set_title('Mean Squared Error (MSE)', fontsize=14)  # Change title font size here
axes[0].set_xlabel('Estimator', fontsize=12)  # Change x-axis label font size here
axes[0].set_ylabel('MSE', fontsize=12)  # Change y-axis label font size here
axes[0].legend()
axes[0].grid(True)  # Enable gridlines for the MSE plot

# Plotting MAE with specified colors and font sizes and adding major gridlines
axes[1].plot(data['Estimator'], data['Train_MAE'], label='Train MAE', color='blue')  # Change line color here
axes[1].plot(data['Estimator'], data['Val_MAE'], label='Validation MAE', color='green', linestyle='--')  # Change line color here
axes[1].set_title('Mean Absolute Error (MAE)', fontsize=14)  # Change title font size here
axes[1].set_xlabel('Estimator', fontsize=12)  # Change x-axis label font size here
axes[1].set_ylabel('MAE', fontsize=12)  # Change y-axis label font size here
axes[1].legend()
axes[1].grid(True)  # Enable gridlines for the MAE plot

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
