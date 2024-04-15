import os
import numpy as np


def text_to_vector(formated_text):
    vec = np.zeros(26)
    for char in formated_text:
        if 'a' <= char.lower() <= 'z':
            index = ord(char) - ord('a')
            vec[index] += 1
    return vec


def format_text(base_text):
    formated_text = base_text.lower()
    letter_vector = text_to_vector(formated_text)
    return letter_vector


def load_training_data(directory):
    X_train, y_train = [], []
    lang_dictionary = {}
    index = 0
    for root, directories, files in os.walk(directory):
        for dir_ in directories:
            lang_dictionary[index] = dir_
            dir_path = os.path.join(root, dir_)
            for file in os.listdir(str(dir_path)):
                file_path = os.path.join(str(dir_path), file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    vec = format_text(f.read())
                    X_train.append(vec)
                    y_train.append(index)
            index += 1
    lang_number = len(np.unique(y_train))
    return np.array(X_train), np.array(y_train), lang_dictionary, lang_number
