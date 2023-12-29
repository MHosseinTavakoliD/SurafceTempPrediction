import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle


# Function to plot all metrics with consistent style across charts
def plot_all_metrics_consistent_style(data, metrics, title):
    # Setting the style
    sns.set(style="whitegrid")

    # Creating subplots
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))

    # Custom color palette
    palette = sns.color_palette("husl", len(data['Model'].unique()))

    # Creating a color and marker iterator
    colors = cycle(palette)
    markers = cycle(['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'])

    # Plotting each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for model in data['Model'].unique():
            model_data = data[data['Model'] == model]
            color = next(colors)
            marker = next(markers)
            ax.plot(model_data['Epoch'], model_data[f'Train {metric}'], label=f'{model} (Train)',
                    color=color, marker=marker, linestyle='-', linewidth=1.5, markersize=5)
            ax.plot(model_data['Epoch'], model_data[f'Validation {metric}'], label=f'{model} (Validation)',
                    color=color, marker=marker, linestyle='--', linewidth=1.5, markersize=5)
        ax.set_title(metric)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)

    # Adjusting layout
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 0.75, 0.95])

    # Adding legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.77, 0.5), fontsize='large')

    # Save the figure
    plt.savefig('model_performance_metrics.png', bbox_inches='tight')

    plt.show()


# Load your data
data = pd.read_csv('LSTMallModelEvaluationHistory.csv')  # Replace with your data file path

# Define the metrics you want to plot
metrics = ['Loss', 'MSE', 'MAE']
title = 'Model Performance Metrics (Train vs Validation)'

# Call the function to plot the metrics
plot_all_metrics_consistent_style(data, metrics, title)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

def plot_all_metrics_consistent_style(data, metrics, title):
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 6))
    models = data['Model'].unique()

    # Predefined colors and markers for consistency across plots
    color_map = {model: color for model, color in zip(models, sns.color_palette("tab10", len(models)))}
    marker_map = {model: marker for model, marker in zip(models, cycle(['o', 's', '^', 'D', 'x', '+']))}
    marker_size = 3  # Consistent smaller marker size
    line_weight = 0.5  # Consistent thinner line weight

    lines = []  # To store legend handles
    labels = []  # To store legend labels

    for i, metric in enumerate(metrics):
        for model in models:
            color = color_map[model]
            marker = marker_map[model]
            model_data = data[data['Model'] == model]
            line_train, = axes[i].plot(model_data['Epoch'], model_data[f'Train {metric}'],
                                       linestyle='-', marker=marker, color=color, markersize=marker_size, linewidth=line_weight)
            line_val, = axes[i].plot(model_data['Epoch'], model_data[f'Validation {metric}'],
                                     linestyle='--', marker=marker, color=color, markersize=marker_size, linewidth=line_weight)
            if i == 0:  # Add to legend only for the first metric
                lines.extend([line_train, line_val])
                labels.extend([f'{model} (Train)', f'{model} (Validation)'])

        axes[i].set_title(f'{metric}')
        axes[i].set_xlabel('Epoch', fontsize=12)  # Increase font size for x-axis labels
        axes[i].set_ylabel(metric, fontsize=12)  # Increase font size for y-axis labels
        axes[i].tick_params(axis='both', which='major', labelsize=10)  # Increase font size for axis numbers

    fig.suptitle(title)
    fig.legend(lines, labels, loc='right', bbox_to_anchor=(1.1, 0.5), fontsize=10)  # Increase font size for legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for legend
    plt.show()

# Load your data
data = pd.read_csv('LSTMallModelEvaluationHistory.csv')  # Replace with your file path

# Plotting all metrics with consistent style across charts
plot_all_metrics_consistent_style(data, ['Loss', 'MSE', 'MAE'], 'Model Performance Metrics (Train vs Validation)')
