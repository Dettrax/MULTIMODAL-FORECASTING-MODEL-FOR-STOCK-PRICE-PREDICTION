import matplotlib.pyplot as plt
import numpy as np

# Data
data = {
    'Single Layer': {
        'No': [2.606, 1.6143, 1.208, 0.899],
        'XGB': [1.96, 1.4, 1.053, 0.911]
    },
    'Multilayer': {
        'No': [4.401, 2.098, 1.565, 0.83],
        'XGB': [1.6534, 1.28, 0.979, 0.933]
    },
    'Bidirectional': {
        'No': [2.89, 1.701, 1.266, 0.88],
        'XGB': [1.938, 1.39, 1.029, 0.927]
    }
}

model_configurations = list(data.keys())
fine_tune_options = ['No', 'XGB']
metrics = ['MSE', 'RMSE', 'MAE', 'R2']
bar_width = 0.35
index = np.arange(len(model_configurations))

# Data information
data_info = {
    'Single Layer': ['P+A+G', 'P+A+G'],
    'Multilayer': ['P+A+G', 'P+A+G'],
    'Bidirectional': ['P+A+G', 'P+A+G']
}

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

for i, metric in enumerate(metrics):
    row = i // 2
    col = i % 2
    ax = axs[row, col]

    for j, fine_tune in enumerate(fine_tune_options):
        bars = ax.bar(index + j * bar_width, [data[model][fine_tune][i] for model in model_configurations], bar_width, label=f'Fine-Tune: {fine_tune}')

    ax.set_xlabel('Model Configuration')
    ax.set_ylabel(metric)
    ax.set_title(metric + ' Comparison')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(model_configurations)
    ax.legend()


    for k, rect in enumerate(bars):
        data_used = data_info[model_configurations[k]][j]
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), data_used, ha='center', va='bottom')

plt.tight_layout()
plt.show()
