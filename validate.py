"""Validate the example problem"""
import pickle
import numpy as np
import train

def classify(i):
    """A really quick nasty way to validate the model"""
    prediction = PICKLE_MODEL.predict(np.array([train.extract_features(i)]))
    result = np.argmax(prediction)
    if result == 1:
        return 'fizz'
    if result == 2:
        return 'buzz'
    if result == 3:
        return 'fizzbuzz'
    return str(i)

if __name__ == "__main__":
    train.make_keras_picklable()
    PKL_FILENAME = "mymodel.pkl"
    with open(PKL_FILENAME, 'rb') as file:
        PICKLE_MODEL = pickle.load(file)
    print([classify(i) for i in range(1, 100 + 1)])
