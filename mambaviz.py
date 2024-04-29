import matplotlib.pyplot as plt

# Data
models = ['Mark1', 'Mark2', 'Mark3']
metrics = ['MSE', 'RMSE', 'MAE', 'R2']
data = {
    'Mark1': {'MSE': 2.454, 'RMSE': 1.566, 'MAE': 1.168, 'R2': 0.905, 'Data': 'P'},
    'Mark2': {'MSE': 1.2202, 'RMSE': 1.1045, 'MAE': 0.8512, 'R2': 0.95148, 'Data': 'P+S'},
    'Mark3': {'MSE': 1.22, 'RMSE': 1.1045, 'MAE': 0.85126, 'R2': 0.9514, 'Data': 'P+S+N'},
    'p+a+g': {'MSE': 2.48, 'RMSE': 1.575, 'MAE': 1.176, 'R2': 0.904, 'Data': 'P+A+G'},
    'p+a+g+s': {'MSE': 1.2155, 'RMSE': 1.102, 'MAE': 0.847, 'R2': 0.9516, 'Data': 'P+A+G+S'},
    'p+a+g+s+n_v': {'MSE': 1.2202, 'RMSE': 1.1045, 'MAE': 0.8512, 'R2': 0.9514, 'Data': 'P+A+G+S+N'}
}

# Plotting first two models
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
fig.suptitle('Selective Space Modelling')

# Collecting metric values from all models except last two
all_metric_values = [data[model][metric] for model in models for metric in metrics]

# Set y-axis limit dynamically
ylim = max(all_metric_values) * 1.1

for i, metric in enumerate(metrics):
    row = i // 2
    col = i % 2
    ax = axes[row, col]

    bars = ax.bar(models, [data[model][metric] for model in models], color=['blue', 'orange', 'green'])
    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.set_ylim(0, ylim)  # Set y-axis limit dynamically

    # Adding data info
    for j, bar in enumerate(bars):
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords='offset points',
                    ha='center', va='bottom')

    # Adding labels
    for j, model in enumerate(models):
        ax.text(j, 0, data[model]['Data'], ha='center', va='bottom')

# Plotting last two models on separate plot
fig2, axes2 = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
fig2.suptitle('Selective Space Modelling')

# Collecting metric values from last two models
all_metric_values_last_two = [data[model][metric] for model in list(data.keys())[3:] for metric in metrics]

# Set y-axis limit dynamically
ylim_last_two = max(all_metric_values_last_two) * 1.1

for i, metric in enumerate(metrics):
    row = i // 2
    col = i % 2
    ax2 = axes2[row, col]

    bars2 = ax2.bar(list(data.keys())[3:], [data[model][metric] for model in list(data.keys())[3:]], color='gray')
    ax2.set_title(metric)
    ax2.set_ylabel(metric)
    ax2.set_ylim(0, ylim_last_two)  # Set y-axis limit dynamically

    # Adding data info
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords='offset points',
                     ha='center', va='bottom')

    # Adding labels
    for j, model in enumerate(list(data.keys())[3:]):
        ax2.text(j, 0, data[model]['Data'], ha='center', va='bottom')

plt.tight_layout()
plt.show()
#save the plot
fig.savefig('Mamba_perf.png')