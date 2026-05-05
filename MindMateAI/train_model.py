import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample dataset
data = {
    "text": [
        "I am very happy today",
        "I feel so sad and lonely",
        "I am stressed because of exams",
        "I am angry at my friend",
        "I feel scared about future"
    ],
    "emotion": [
        "joy",
        "sadness",
        "stress",
        "anger",
        "fear"
    ]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["emotion"]

model = LogisticRegression()
model.fit(X, y)

pickle.dump(model, open("emotion_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained successfully!")