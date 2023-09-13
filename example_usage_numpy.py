import numpy as np
import matplotlib.pyplot as plt
from oja_algorithm_numpy import Oja

# Instantiate Oja class
ojanet = Oja(minimized_data_size=1, step=0.01)

# Define some input data
data = np.array([[2, 2], [1, 1], [4, 4], [5, 5]])

# Train the model on the data for 100 epochs
maes = ojanet.train(data, epochs=100)

# Plot MAE vs Epochs
plt.figure(figsize=(10, 6))
plt.plot(maes, marker='o', linestyle='-', color='b')
plt.title('Mean Absolute Error (MAE) Over Epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Mean Absolute Error (MAE)', fontsize=14)
plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('oja_mae_vs_epochs.png', dpi=300)
plt.show()
