"""Validate the example problem"""
import pickle
import numpy as np
import train
import keras
import unittest


class TestAccuracy(unittest.TestCase):
    PKL_FILENAME = "mymodel.pkl"

    def test_sufficient_accuracy(self):
        train.make_keras_picklable()    
        with open(self.PKL_FILENAME, 'rb') as file:
            model = pickle.load(file)
        training_points = range(0, 101)
        data = np.array([train.extract_features(i) for i in training_points])
        labels = np.array([train.fizzbuzz(i) for i in training_points])
        score = model.evaluate(data, keras.utils.to_categorical(labels), batch_size=64)

        self.assertGreaterEqual(score[1], 0.95)

if __name__ == "__main__":
    unittest.main()
