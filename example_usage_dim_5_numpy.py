
import numpy as np
import matplotlib.pyplot as plt
from oja_algorithm_numpy import Oja

# Instantiate Oja class with minimized_data_size set to 2
ojanet = Oja(minimized_data_size=2, step=0.01)

# Define a new input data with more dimensions (5 dimensions in this case)
data = np.array([
    [1, 2, 3, 4, 5],
    [2, 4, 6, 8, 10],
    [3, 6, 9, 12, 15],
    [4, 8, 12, 16, 20],
    [5, 10, 15, 20, 25]
])

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
