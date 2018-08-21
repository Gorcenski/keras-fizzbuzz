import unittest


class TestAccuracy(unittest.TestCase):
    METRICS_FILE = "accuracy.txt"

    def test_95percent_accuracy(self):    
        with open(self.METRICS_FILE, 'r') as file:
            accuracy = float(file.read())

        self.assertGreaterEqual(accuracy, 0.95)


if __name__ == "__main__":
    unittest.main()