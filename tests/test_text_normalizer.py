import unittest
from video_assembler.services.alignment.text_normalizer import TextNormalizer

class TestTextNormalizer(unittest.TestCase):
    def setUp(self):
        self.normalizer = TextNormalizer()

    def test_lowercase(self):
        self.assertEqual(self.normalizer.normalize("HELLO World"), ["hello", "world"])

    def test_punctuation(self):
        self.assertEqual(self.normalizer.normalize("Hello, world! How's it going?"), ["hello", "world", "hows", "it", "going"])

    def test_contractions(self):
        # wanna -> want to, don't -> do not
        self.assertEqual(self.normalizer.normalize("you want to own a truck"), ["you", "wanna", "own", "a", "truck"])
        self.assertEqual(self.normalizer.normalize("i am going to do it"), ["im", "gonna", "do", "it"])
        
    def test_numbers(self):
        # 10 -> ten
        self.assertEqual(self.normalizer.normalize("I have 10 apples"), ["i", "have", "ten", "apples"])
        self.assertEqual(self.normalizer.normalize("cost is $5"), ["cost", "is", "five", "dollars"])

if __name__ == '__main__':
    unittest.main()
