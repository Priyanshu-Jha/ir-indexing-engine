# core.py
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

class TextProcessor:
    """Handles all text cleaning and normalization."""
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def process(self, text):
        """
        Tokenizes, cleans, removes stopwords, and stems the text.
        Returns a list of processed tokens.
        """
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = word_tokenize(text)
        return [self.stemmer.stem(word) for word in tokens if word not in self.stop_words]