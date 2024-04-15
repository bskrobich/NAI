from data_util import *
from perceptrons_trainer import *
from input_classifier import *


def main():
    data_directory = "training_data/"
    X, y, languages, lang_number = load_training_data(data_directory)
    perceptrons = train_perceptrons(X, y, lang_number)
    classify_input(perceptrons, languages)


if __name__ == "__main__":
    main()
