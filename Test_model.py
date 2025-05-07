import unittest
import main

class TestFakeNewsModel(unittest.TestCase):
    def setUp(self):
        self.model, self.vectorizer = main.load_model()

    def test_real_news(self):
        text = "The president held a press conference today."
        vec = self.vectorizer.transform([text])
        prediction = self.model.predict(vec)[0]
        self.assertIn(prediction, ['REAL', 'FAKE'])

    def test_empty_string(self):
        text = ""
        vec = self.vectorizer.transform([text])
        prediction = self.model.predict(vec)[0]
        self.assertIn(prediction, ['REAL', 'FAKE'])

if __name__ == '__main__':
    unittest.main()
