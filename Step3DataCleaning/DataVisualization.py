import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# Load your data
df = pd.read_csv('/Archive/DBV1Wis23.csv', encoding='ISO-8859-1')

# Convert 'MeasureTime' to datetime if it's not already
df['MeasureTime'] = pd.to_datetime(df['MeasureTime'])

# Extract the month and create a new column 'Month'
df['Month'] = df['MeasureTime'].dt.month

# Select only numeric columns for the correlation matrix, including the new 'Month' column
numeric_df = df.select_dtypes(include=[np.number])

# Calculate the correlation matrix
corr_matrix = numeric_df.corr()

# Plot the heatmap
# Increase the figure size
plt.figure(figsize=(20, 16))  # You can adjust these values as needed

# Generate the heatmap
ax = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')

# Rotate the x-axis labels if necessary
plt.xticks(rotation=45, ha='right', fontsize=10)  # Adjust fontsize as needed

# Rotate the y-axis labels if necessary
plt.yticks(rotation=0, fontsize=10)  # Adjust fontsize as needed

# Ensure the labels on the axes are fully visible
plt.tight_layout()

# Set a title
plt.title('Correlation Matrix Heatmap', fontsize=14)  # Adjust fontsize as needed

# Show the plot
plt.show()

