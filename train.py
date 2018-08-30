"""Generate training data for a sample problem and train a model on it"""
import tempfile
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

def make_keras_picklable():
    """Took this from the internet, don't really understand what it is, tbh"""
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as f_d:
            keras.models.save_model(self, f_d.name, overwrite=True)
            model_str = f_d.read()
        return {'model_str': model_str}

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as f_d:
            f_d.write(state['model_str'])
            f_d.flush()
            model = keras.models.load_model(f_d.name)
        self.__dict__ = model.__dict__


    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


# These methods are just for generating example data --
# normally we'd be loading this in
def extract_features(i):
    """Train on the prime modulii of the number for primes under 15"""
    return [i % 2, i % 3, i % 5, i % 7, i % 11, i % 13]


def fizzbuzz(i):
    """Canonical fizzbuzz solution to generate category labels"""
    if i % 15 == 0:
        return 3
    if i % 5 == 0:
        return 2
    if i % 3 == 0:
        return 1
    return 0


def generate_data(support):
    """Generate training data"""
    training_points = range(support[0], support[1])
    data = np.array([extract_features(i) for i in training_points])
    labels = np.array([fizzbuzz(i) for i in training_points])

    # Convert labels to categorical one-hot encoding which is required by Keras
    # the labels must be ordinal numbers starting with 0
    return data, keras.utils.to_categorical(labels)


def compile_model(data):
    """Set up the model based on the data"""
    dim = np.shape(data)[1]
    model = Sequential()
    model.add(Dense(10, activation='relu', input_dim=dim))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(model, data, labels):
    """Train the model using the generated data/labels"""
    model.fit(data, labels, epochs=200, batch_size=64, shuffle=True)
    score = model.evaluate(data, labels, batch_size=64)
    print(score)
    print(model.metrics_names)
    pkl_filename = "mymodel.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)


if __name__ == "__main__":
    make_keras_picklable()
    RAW_DATA, ONE_HOT_LABELS = generate_data([500, 5500])
    train_model(compile_model(RAW_DATA), RAW_DATA, ONE_HOT_LABELS)
