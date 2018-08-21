"""Validate the example problem"""
import pickle
import numpy as np
import keras
import train


PKL_FILENAME = "mymodel.pkl"

def evaluate():
    train.make_keras_picklable()
    with open(PKL_FILENAME, 'rb') as file:
        model = pickle.load(file)
    training_points = range(0, 101)
    data = np.array([train.extract_features(i) for i in training_points])
    labels = np.array([train.fizzbuzz(i) for i in training_points])
    score = model.evaluate(data, keras.utils.to_categorical(labels), batch_size=64)
    with open("accuracy.txt", 'w') as file:
        file.write(str(score[1]))


if __name__ == "__main__":
    evaluate()
