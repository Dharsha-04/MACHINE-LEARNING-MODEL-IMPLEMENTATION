# ==============================
# SPAM EMAIL DETECTION PROJECT
# ==============================

import pandas as pd
import numpy as np
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB

# Download stopwords
nltk.download('stopwords')

# ==============================
# LOAD DATASET
# ==============================

print("Loading dataset...")

df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only required columns
df = df[['v1','v2']]

# Rename columns
df.columns = ['label','message']

print("Dataset loaded successfully")
print(df.head())


# ==============================
# DATA VISUALIZATION
# ==============================

print("Visualizing dataset...")

sns.countplot(x='label', data=df)
plt.title("Spam vs Ham Emails")
plt.show()


# ==============================
# TEXT CLEANING FUNCTION
# ==============================

stop_words = set(stopwords.words('english'))

def clean_text(text):

    text = text.lower()

    text = ''.join([c for c in text if c not in string.punctuation])

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)


df["clean_message"] = df["message"].apply(clean_text)

print("Text cleaned successfully")


# ==============================
# FEATURE EXTRACTION
# ==============================

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["clean_message"])

y = df["label"]

print("Text converted to numerical features")


# ==============================
# TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Data split completed")


# ==============================
# TRAIN MODEL
# ==============================

print("Training model...")

model = MultinomialNB()

model.fit(X_train, y_train)

print("Model training completed")


# ==============================
# PREDICTION
# ==============================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)


# ==============================
# CLASSIFICATION REPORT
# ==============================

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))


# ==============================
# CONFUSION MATRIX
# ==============================

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ==============================
# CUSTOM EMAIL TEST
# ==============================

def predict_spam(text):

    text = clean_text(text)

    vector = vectorizer.transform([text])

    prediction = model.predict(vector)

    return prediction[0]


test_email = "Congratulations! You won a free prize"

result = predict_spam(test_email)

print("\nTest Email:", test_email)
print("Prediction:", result)

print("\nProject Completed Successfully!")