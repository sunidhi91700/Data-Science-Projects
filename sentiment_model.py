import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.model = MultinomialNB()

    def load_data_from_txt(self, filepath):
        reviews, labels = [], []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.match(r"__label__([12])\s+(.*)", line)
                if match:
                    label = 'positive' if match.group(1) == '2' else 'negative'
                    review = match.group(2).strip()
                    labels.append(label)
                    reviews.append(review)
        return reviews, labels

    def train(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)

    def predict(self, text):
        X = self.vectorizer.transform([text])
        return self.model.predict(X)[0]

    def save(self):
        joblib.dump(self.model, 'nb_model.pkl')
        joblib.dump(self.vectorizer, 'vectorizer.pkl')

    def load(self):
        self.model = joblib.load('nb_model.pkl')
        self.vectorizer = joblib.load('vectorizer.pkl')