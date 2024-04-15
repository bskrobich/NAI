import numpy as np

from data_util import format_text
from vector_util import normalize_vector


def classify_input(perceptrons, languages):
    while True:
        user_input = input("Enter text to classify or (Q/q)' to exit: ")
        if user_input.lower() in {'Q', 'q'}:
            break
        input_vector = format_text(user_input)
        normalized_vector = normalize_vector(input_vector)

        outputs = []
        for perceptron in perceptrons:
            output = perceptron.predict(normalized_vector)
            outputs.append(output)
        max_output_index = np.argmax(outputs)
        print("Predicted language:", languages[max_output_index])