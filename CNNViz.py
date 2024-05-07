import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame from the provided data
data = {
    'MSE': [2.821, 2.38, 4.221, 1.9359, 4.9, 5.085, 0.921, 1.293, 3.0087, 2.851, 1.0775, 1.436],
    'RMSE': [1.679, 1.5436, 2.054, 1.391, 2.21, 2.255, 0.959, 1.137, 1.7345, 1.688, 1.038, 1.198],
    'MAE': [1.2499, 1.2092, 1.5709, 1.017, 1.698, 1.7402, 0.7521, 0.8512, 1.306162, 1.2706, 0.797, 0.8967],
    'R2': [0.891, 0.9174, 0.8372, 0.932, 0.81, 0.80392, 0.964, 0.9483, 0.8839, 0.89, 0.9513, 0.9573],
    'Data': ['Price', 'Price+Arima+Garch', 'Price+Arima+Garch', 'Price', 'Price+Arima+Garch', 'Price', 'Price', 'Price+Arima+Garch', 'Price+Arima+Garch', 'Price', 'Price', 'Price+Arima+Garch'],
    'FINE-TUNE': ['No', 'Soft', 'No', 'XGB', 'No', 'No', 'XGB', 'XGB', 'No', 'XGB', 'No', 'XGB'],
    'Type': ['Soft', 'Soft', 'Soft', 'Soft', 'Scaled-Dot', 'Scaled-Dot', 'Scaled-Dot', 'Scaled-Dot', 'Multi-head', 'Multi-head', 'Multi-head', 'Multi-head']
}

df = pd.DataFrame(data)

# Separate the data based on 'FINE-TUNE'
fine_tune_yes = df[df['FINE-TUNE'] != 'No']
fine_tune_no = df[df['FINE-TUNE'] == 'No']

# Plotting separate graphs for 'FINE-TUNE' Yes and No
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

for ax, subset, title in zip(axes, [fine_tune_yes, fine_tune_no], ['Fine-Tuned', 'Not Fine-Tuned']):
    ax.scatter(subset.index, subset['RMSE'], label='RMSE')
    ax.scatter(subset.index, subset['MAE'], label='MAE')
    ax.scatter(subset.index, subset['R2'], label='R2')
    ax.set_xticks(subset.index)
    ax.set_xticklabels(subset['Type'], rotation=45)
    ax.set_title(title)
    ax.legend()

    # Annotate each point with its R2 value
    for i, txt in enumerate(subset['R2']):
        ax.annotate(f"{txt:.3f}", (subset.index[i], subset['R2'].iloc[i]))

plt.tight_layout()
plt.show()
