import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Download resource NLTK (kalau belum pernah di download)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Fungsi preprocessing teks
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Hapus karakter selain huruf dan spasi
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenisasi dan hapus stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

# Baca data dari CSV
df = pd.read_csv('restaurant_reviews.csv')

print("Data awal:")
print(df.head())

# Preprocess review text
df['clean_review'] = df['review'].apply(preprocess_text)

print("\nSetelah preprocessing:")
print(df[['review', 'clean_review']].head())

# Ubah label sentiment ke angka (Positive=1, Negative=0)
df['label'] = df['sentiment'].map({'Positive': 1, 'Negative': 0})

# Pisahkan fitur dan label
X = df['clean_review']
y = df['label']

# Bagi data training dan testing (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ubah teks ke fitur numerik dengan TF-IDF
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Latih model Naive Bayes
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Prediksi data testing
y_pred = model.predict(X_test_tfidf)

# Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAkurasi: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Visualisasi Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
