import numpy as np
import os
from perceptron import Perceptron


def get_training_data(directory):
    X_train, y_train = [], []
    lang_dictionary = {}
    index = 0

    for root, directories, files in os.walk(directory):
        for d in directories:
            lang_dictionary[index] = d
            dir_path = os.path.join(root, d)
            for file in os.listdir(str(dir_path)):
                file_path = os.path.join(str(dir_path), file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    filtered_text = ''.join(filter(str.isalpha, f.read())).lower()
                    vec = text_to_vector(filtered_text)
                    X_train.append(vec)
                    y_train.append(index)
            index += 1
    return np.array(X_train), np.array(y_train), lang_dictionary


def text_to_vector(filtered_text):
    vec = np.zeros(26)
    for char in filtered_text:
        if 'a' <= char.lower() <= 'z':
            index = ord(char.lower()) - ord('a')
            vec[index] += 1
    return vec


def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vector
    return vector / norm


def normalize_weights(weights):
    norm = np.linalg.norm(weights)
    if norm == 0:
        return weights
    return weights / norm


def format_text(plain_text):
    filtered_text = ''.join(filter(str.isalpha, plain_text)).lower()
    vec = text_to_vector(filtered_text)
    return vec


if __name__ == "__main__":
    data_directory = "data/train"
    X, y, languages = get_training_data(data_directory)
    n_languages = len(np.unique(y))

    print(n_languages)

    perceptrons = []
    for i in range(n_languages):
        perceptron = Perceptron(learning_rate=0.5, epochs=2000)
        y_cls = np.where(y == i, 1, 0)
        perceptron.fit(X, y_cls)
        perceptrons.append(perceptron)

    while True:
        text = input("Enter text to classify (or 'quit' to exit): ")
        if text.lower() == "quit":
            break
        vector = format_text(text)
        normalized_vector = normalize_vector(vector)

        outputs = []
        for perceptron in perceptrons:
            output = perceptron.predict(normalized_vector)
            outputs.append(output)

        max_output_index = np.argmax(outputs)
        print("Predicted language:", languages[max_output_index])
