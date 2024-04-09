import numpy as np

from perceptron import Perceptron


def check_label(label, label_dictionary):
    if label not in {"0", "1"}:
        if not label_dictionary:
            label_dictionary[label] = 1
            return 1
        else:
            if label in label_dictionary:
                return label_dictionary[label]
            else:
                label_dictionary[label] = 0
                return 0
    else:
        return label


def get_data(file_path, label_dictionary):
    try:
        file = open(file_path)
        features_list = []
        labels_list = []

        for line in file:
            features = line.split(",")[:-1]
            label = line.split(",")[-1].rstrip("\n")
            features_list.append(features)
            labels_list.append(check_label(label, label_dictionary))

        return np.array(features_list, dtype=float), np.array(labels_list, dtype=int)
    except FileNotFoundError as e:
        print(e)


def show_UI():
    try:
        pressed_key = None
        training_path = input("Enter training data path: ")
        test_path = input("Enter test data path: ")
        learning_rate = float(input("Enter learning rate between 0 and 1: "))
        epochs = int(input("Enter number of epochs: "))
        print()

        label_dictionary = {}

        X_train, y_train = get_data(training_path, label_dictionary)
        X_test, y_test = get_data(test_path, label_dictionary)

        perceptron = Perceptron(X_test, y_test, learning_rate=learning_rate, epochs=epochs)
        perceptron.fit(X_train, y_train)

        while pressed_key != "q":
            print("\nPRESS (1) TO ENTER A NEW OBSERVATION AND PREDICT THE CLASS.")
            print("PRESS (Q/q) TO EXIT.")
            pressed_key = input("-> PRESSED KEY: ")
            if pressed_key == "1":
                user_input = input("\nEnter the observation, each feature after ',': ")
                features_list = np.array(user_input.split(","), dtype=float)
                res = perceptron.predict(features_list)
                print("PREDICTED CLASS IS {}.".format(res))
                pass
            elif pressed_key in ["Q", "q"]:
                break
            else:
                print("Command not found")

    except (ValueError, Exception):
        print("Error occurred")


if __name__ == '__main__':
    show_UI()
