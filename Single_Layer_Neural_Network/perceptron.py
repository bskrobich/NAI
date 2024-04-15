import numpy as np


class Perceptron:
    def __init__(self, X_test=None, y_test=None, learning_rate=0.01, epochs=1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.threshold = None
        self.X_test, self.y_test = X_test, y_test

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.random.rand(n_features) * 0.1
        self.threshold = 0.05

        for epoch in range(self.epochs):
            for index, x in enumerate(X):
                net = np.dot(x, self.weights) - self.threshold
                predicted_output = self.activation_function(net)
                if predicted_output != y[index]:
                    self.weights += (y[index] - predicted_output) * self.learning_rate * x
                    self.threshold += (y[index] - predicted_output) * self.learning_rate * (-1)

    def predict(self, X):
        return np.dot(X, self.weights) - self.threshold

    @staticmethod
    def activation_function(net):
        return np.where(net >= 0, 1, 0)
