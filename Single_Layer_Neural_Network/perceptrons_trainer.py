import numpy as np
from perceptron import Perceptron


def train_perceptrons(X, y, lang_num):
    perceptrons = []
    for i in range(lang_num):
        perceptron = Perceptron(learning_rate=0.5, epochs=2000)
        bin_label = np.where(y == i, 1, 0)
        perceptron.fit(X, bin_label)
        perceptrons.append(perceptron)
    return perceptrons
