# Import library yang dibutuhkan
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Download stopwords dan wordnet untuk NLTK
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Buat data dummy
data = {
    'Review': [
        "The food was absolutely wonderful, from preparation to presentation, very pleasing.",
        "I hated the service, the waiter was very rude and slow.",
        "What a great experience! The staff was friendly and the food was delicious.",
        "Terrible! The food was cold and tasteless, will not come back.",
        "Delicious food and cozy atmosphere. Highly recommend!",
        "The restaurant was dirty and the food was bad.",
        "Excellent place, loved the desserts and the service was excellent too.",
        "Worst experience ever, food took forever to arrive and it was cold.",
        "Amazing flavors and fresh ingredients, will visit again.",
        "Not worth the price, poor service and bad food."
    ],
    'Sentiment': [
        'Positive',
        'Negative',
        'Positive',
        'Negative',
        'Positive',
        'Negative',
        'Positive',
        'Negative',
        'Positive',
        'Negative'
    ]
}

# Load data ke DataFrame
df = pd.DataFrame(data)

# Lihat data
print("Data awal:")
print(df.head(), "\n")

# Preprocessing text
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # hapus tanda baca/non-alfabet
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

df['cleaned_review'] = df['Review'].apply(preprocess_text)

print("Setelah preprocessing:")
print(df[['Review', 'cleaned_review']], "\n")

# Encode label Sentiment menjadi angka
le = LabelEncoder()
df['label'] = le.fit_transform(df['Sentiment'])  # Positive=1, Negative=0

# Feature extraction menggunakan TF-IDF
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(df['cleaned_review']).toarray()
y = df['label']

# Split data train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training model Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict di test set
y_pred = model.predict(X_test)

# Evaluasi model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Visualisasi confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
