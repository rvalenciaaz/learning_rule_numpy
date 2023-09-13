import numpy as np

class Oja:
    def __init__(self, minimized_data_size=1, step=0.01, weight=None):
        self.minimized_data_size = minimized_data_size
        self.step = step
        self.weight = weight

    def _initialize_weights(self, X):
        """Initialize weights with normal distribution."""
        if self.weight is None:
            self.weight = np.random.normal(size=(X.shape[1], self.minimized_data_size))

    def one_training_update(self, X):
        """
        Performs one training update using Oja's rule.
        
        Parameters:
        X (numpy.ndarray): The input data matrix with shape (n_samples, n_features)
        
        Returns:
        float: The mean absolute error (MAE) for this training step
        """
        self._initialize_weights(X)
        
        minimized = np.dot(X, self.weight)
        reconstruct = np.dot(minimized, self.weight.T)
        error = X - reconstruct

        self.weight += self.step * (np.dot(X.T, minimized) - np.sum(minimized**2, axis=0) * self.weight)
        mae = np.mean(np.abs(error))

        return mae

    def train(self, X, epochs=100):
        """Trains the model using the given data and number of epochs."""
        maes = []
        for _ in range(epochs):
            mae = self.one_training_update(X)
            maes.append(mae)
        return maes

    def reconstruct(self, X):
        """Reconstructs the original data from the minimized data."""
        self._initialize_weights(X)
        minimized = np.dot(X, self.weight)
        return np.dot(minimized, self.weight.T)

    def predict(self, X):
        """Applies dimensionality reduction to the given data."""
        self._initialize_weights(X)
        return np.dot(X, self.weight)
