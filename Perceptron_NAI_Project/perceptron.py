import numpy as np


def accuracy(y_true, y_pred):
    acc = np.mean(y_pred == y_true)
    return acc


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
            combined = list(zip(X, y))
            np.random.shuffle(combined)
            X_shuffled, y_shuffled = zip(*combined)
            X_shuffled = np.array(X_shuffled)
            y_shuffled = np.array(y_shuffled)

            for index, x in enumerate(X_shuffled):
                net = np.dot(x, self.weights) - self.threshold
                predicted_output = self.activation_function(net)
                if predicted_output != y_shuffled[index]:
                    self.weights += (y_shuffled[index] - predicted_output) * self.learning_rate * x
                    self.threshold += (y_shuffled[index] - predicted_output) * self.learning_rate * (-1)
            prediction = self.predict(self.X_test)
            print("Epoch no.",  str(epoch + 1) + ", accuracy:", accuracy(self.y_test, prediction))

    def predict(self, X):
        net = np.dot(X, self.weights) - self.threshold
        predicted_output = self.activation_function(net)
        return predicted_output

    @staticmethod
    def activation_function(net):
        return np.where(net >= 0, 1, 0)
