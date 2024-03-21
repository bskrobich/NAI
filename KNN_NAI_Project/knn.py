import numpy as np
from collections import Counter


def get_euclidean_distance(x_test, x_train):
    return np.sqrt(np.sum((x_test - x_train) ** 2))


class KNN:
    def __init__(self, k):
        self.k = k
        self.X_train_features = None
        self.y_train_labels = None
        self.count = 0

    # Set the number of nearest neighbors
    def set_k(self, k):
        self.k = k

    # Store TRAINING samples
    def fit(self, X, y):
        self.X_train_features = X
        self.y_train_labels = y

    # This method receives multiple TEST samples
    def predict(self, X):
        predict_labels = [self.predict_sample(x) for x in X]
        return np.array(predict_labels)

    # This method receives single TEST sample
    def predict_sample(self, x_test):
        distances = [get_euclidean_distance(x_test, x_train) for x_train in self.X_train_features]
        k_n_indexes = np.argsort(distances)[:self.k]
        k_n_labels = [self.y_train_labels[i] for i in k_n_indexes]
        self.count += 1
        print(str(self.count) + ": " + Counter(k_n_labels).most_common(1)[0][0])
        return Counter(k_n_labels).most_common(1)[0][0]
