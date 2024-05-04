import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
import numpy as np

# Function to load data from .jsonl file
def load_data(filepath):
    texts = []
    intents = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            texts.append(data['sentence'])  # Accessing the text data
            intents.append(data['intent'])  # Accessing the intent data
    return pd.DataFrame({'text': texts, 'intent': intents})

# Load training and testing data
train_df = load_data('train.jsonl')
test_df = load_data('test.jsonl')

# Prepare the dataset
X_train, y_train = train_df['text'], train_df['intent']
X_test, y_test = test_df['text'], test_df['intent']

# Compute class weights for handling class imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {label: weight for label, weight in zip(np.unique(y_train), class_weights)}

# Define a pipeline combining a text feature extractor with SVM classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.5, ngram_range=(1, 1))),
    ('svm', SVC(kernel='linear', C=10, gamma=0.001, probability=True, class_weight=class_weight_dict))
])

# Train the model
pipeline.fit(X_train, y_train)

# Predictions on test set
y_pred = pipeline.predict(X_test)

# Display the overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall accuracy of the model on the test set: {accuracy:.2f}")

# Classification function with fallback for low confidence
def classify_intent(text):
    pred_proba = pipeline.predict_proba([text])[0]
    max_proba = max(pred_proba)

    if max_proba > 0.7:
        return pipeline.predict([text])[0], max_proba
    else:
        return "NLU fallback: Intent could not be confidently determined", max_proba

# Test the model with test data and print individual results
for index, row in test_df.iterrows():
    intent, confidence = classify_intent(row['text'])
    print(f"Input: {row['text']} - Prediction: {intent}, Confidence: {confidence:.2f}")
