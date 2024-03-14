import numpy as np
from knn import KNN


def get_data_set(f_path):
    features_list = []
    labels_list = []

    file = open(f_path)

    for line in file:
        features = line.split(",")[0:-1]
        label = line.split(",")[-1].rstrip("\n")
        features_list.append(features)
        labels_list.append(label)
    return np.array(features_list, dtype=float), np.array(labels_list)


def user_interface():
    pressed_key = None
    train_path = input("Enter the path to the training file: ")
    global k
    k = int(input("Enter the number (k) of nearest neighbors: "))
    print()

    options = [
        "PRESS (1) TO CLASSIFY ALL OBSERVATIONS IN THE TEST SET FROM FILE.",
        "PRESS (2) TO CLASSIFY GIVEN OBSERVATION.",
        "PRESS (3) TO CHANGE K.",
        "PRESS (Q/q) TO EXIT THE PROGRAM."
    ]
    while pressed_key not in ["Q", "q"]:
        for option in options:
            print(option)
        pressed_key = input("-> PRESSED KEY: ")
        print()

        global classifier, X_train_features, y_train_labels
        classifier = KNN(k=k)
        X_train_features, y_train_labels = get_data_set(train_path)

        if pressed_key == str(1):
            test_path = input("Enter the path to the test file: ")
            print()
            X_test_features, y_test_labels = get_data_set(test_path)

            classifier.fit(X_train_features, y_train_labels)
            pred = classifier.predict_samples(X_test_features)

            acc = float(np.sum(pred == y_test_labels) / len(y_test_labels)) * 100
            print("Accuracy is: ", acc, "%")

            print()
        elif pressed_key == str(2):
            user_obs = input("Enter the observation, each feature after ',': ")
            features_list = np.array(user_obs.split(","), dtype=float)

            classifier.fit(X_train_features, y_train_labels)
            pred = classifier.predict_sample(features_list)
            print()
        elif pressed_key == str(3):
            k = int(input("Enter the new (k) number of nearest neighbors: "))
            print()
        elif pressed_key in ["Q", "q"]:
            break
        else:
            print("OPTION NOT FOUND\n")


if __name__ == "__main__":

    user_interface()