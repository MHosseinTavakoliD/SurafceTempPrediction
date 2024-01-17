# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from itertools import cycle
#
#
# # Function to plot all metrics with consistent style across charts
# def plot_all_metrics_consistent_style(data, metrics, title):
#     # Setting the style
#     sns.set(style="whitegrid")
#
#     # Creating subplots
#     fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))
#
#     # Custom color palette
#     palette = sns.color_palette("husl", len(data['Model'].unique()))
#
#     # Creating a color and marker iterator
#     colors = cycle(palette)
#     markers = cycle(['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'])
#
#     # Plotting each metric
#     for i, metric in enumerate(metrics):
#         ax = axes[i]
#         for model in data['Model'].unique():
#             model_data = data[data['Model'] == model]
#             color = next(colors)
#             marker = next(markers)
#             ax.plot(model_data['Epoch'], model_data[f'Train {metric}'], label=f'{model} (Train)',
#                     color=color, marker=marker, linestyle='-', linewidth=1, markersize=4)
#             ax.plot(model_data['Epoch'], model_data[f'Validation {metric}'], label=f'{model} (Validation)',
#                     color=color, marker=marker, linestyle='--', linewidth=1, markersize=4)
#         ax.set_title(metric)
#         ax.set_xlabel('Epoch')
#         ax.set_ylabel(metric)
#
#     # Adjusting layout
#     fig.suptitle(title, fontsize=16)
#     fig.tight_layout(rect=[0, 0, 0.75, 0.95])
#
#     # Adding legend
#     handles, labels = ax.get_legend_handles_labels()
#     fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1, 0.5), fontsize=14)
#
#     # Save the figure
#     plt.savefig('model_performance_metrics.png', bbox_inches='tight')
#
#     plt.show()
#
#
# # Load your data
# data = pd.read_csv('LSTMallModelEvaluationHistory.csv')  # Replace with your data file path
#
# # Define the metrics you want to plot
# metrics = [ 'MSE', 'MAE']
# title = 'Model Performance Metrics (Train vs Validation)'
#
# # Call the function to plot the metrics
# plot_all_metrics_consistent_style(data, metrics, title)
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from itertools import cycle
#
# def plot_all_metrics_consistent_style(data, metrics, title):
#     fig, axes = plt.subplots(1, len(metrics), figsize=(18, 6))
#     models = data['Model'].unique()
#
#     # Predefined colors and markers for consistency across plots
#     color_map = {model: color for model, color in zip(models, sns.color_palette("tab10", len(models)))}
#     marker_map = {model: marker for model, marker in zip(models, cycle(['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']))}
#     marker_size = 4  # Consistent smaller marker size
#     line_weight = 0.75  # Consistent thinner line weight
#
#     lines = []  # To store legend handles
#     labels = []  # To store legend labels
#
#     for i, metric in enumerate(metrics):
#         for model in models:
#             color = color_map[model]
#             marker = marker_map[model]
#             model_data = data[data['Model'] == model]
#             line_train, = axes[i].plot(model_data['Epoch'], model_data[f'Train {metric}'],
#                                        linestyle='-.', marker=marker, color=color, markersize=marker_size, linewidth=line_weight)
#             line_val, = axes[i].plot(model_data['Epoch'], model_data[f'Validation {metric}'],
#                                      linestyle='--', marker=marker, color=color, markersize=marker_size, linewidth=line_weight)
#             if i == 0:  # Add to legend only for the first metric
#                 lines.extend([line_train, line_val])
#                 labels.extend([f'{model} (Train)', f'{model} (Validation)'])
#
#         axes[i].set_title(f'{metric}')
#         axes[i].set_xlabel('Epoch', fontsize=12)  # Increase font size for x-axis labels
#         axes[i].set_ylabel(metric, fontsize=12)  # Increase font size for y-axis labels
#         axes[i].tick_params(axis='both', which='major', labelsize=10)  # Increase font size for axis numbers
#
#     fig.suptitle(title)
#     fig.legend(lines, labels, loc='center right', bbox_to_anchor=(1, 0.5), fontsize=14)  # Increase font size for legend
#     plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for legend
#     plt.show()
#
# # Load your data
# data = pd.read_csv('LSTMallModelEvaluationHistory.csv')  # Replace with your file path
#
# # Plotting all metrics with consistent style across charts
# plot_all_metrics_consistent_style(data, [ 'MSE', 'MAE'], 'Model Performance Metrics (Train vs Validation)')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
fontsize = 12
def plot_all_metrics_consistent_style(data, metrics, title):
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 6))
    models = data['Model'].unique()

    # Predefined colors and markers for consistency across plots
    color_map = {model: color for model, color in zip(models, sns.color_palette(n_colors=len(models)))}
    marker_map = {model: marker for model, marker in zip(models, cycle(['|', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'X']))}
    line_weight = 0.75  # Consistent thinner line weight

    model_lines = []  # To store model legend handles
    model_labels = []  # To store model legend labels

    for i, metric in enumerate(metrics):
        for model in models:
            color = color_map[model]
            marker = marker_map[model]
            model_data = data[data['Model'] == model]
            line, = axes[i].plot(model_data['Epoch'], model_data[f'Train {metric}'],
                                 linestyle='-.', marker=marker, color=color, markersize=6, linewidth=line_weight, markevery=10)
            axes[i].plot(model_data['Epoch'], model_data[f'Validation {metric}'],
                         linestyle='dotted', marker=marker, color=color, markersize=1, linewidth=line_weight, markevery=10)
            if i == 0:  # Add to model legend only for the first metric
                model_lines.append(line)
                model_labels.append(model)

        axes[i].set_title(f'{metric}')
        axes[i].set_xlabel('Epoch', fontsize=fontsize)
        axes[i].set_ylabel(metric,fontsize=fontsize)
        axes[i].tick_params(axis='both', which='major')

    # Create the model legend at the upper center of the plot
    model_legend = fig.legend(handles=model_lines, labels=model_labels, loc='upper center',
                              bbox_to_anchor=(0.31, .79), ncol=3, frameon=False)

    # Ensure that the style legend is below the model legend
    # This is the part we adjust by changing the bbox_to_anchor
    train_line = plt.Line2D([0], [0], color='black', linestyle='-.', linewidth=line_weight)
    val_line = plt.Line2D([0], [0], color='black', linestyle='--', linewidth=line_weight)
    style_legend = fig.legend(handles=[train_line, val_line], labels=['Train', 'Validation'],
                              loc='upper center', bbox_to_anchor=(0.4, 0.65), ncol=2, frameon=True)
    style_legend.get_frame().set_facecolor('white')
    style_legend.get_frame().set_edgecolor('black')

    # Place the model legend back in the figure
    fig.add_artist(model_legend)

    fig.suptitle(title, y=1.05)
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.show()

# Load your data
data = pd.read_csv('LSTMallModelEvaluationHistory.csv')  # Replace with your file path

# Plotting all metrics with consistent style across charts
plot_all_metrics_consistent_style(data, ['MSE', 'MAE'], 'Model Performance Metrics (Train vs Validation)')


