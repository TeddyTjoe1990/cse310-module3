import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Download resource NLTK (kalau belum pernah di download)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Fungsi preprocessing teks
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

# Buat data dummy besar (contoh 120 data)
positive_reviews = [
    "Amazing food and great service!",
    "Loved the cozy atmosphere and delicious dishes.",
    "Best restaurant in town, highly recommend.",
    "Fantastic flavors and friendly staff.",
    "Excellent experience, will come back for sure.",
    "The desserts were heavenly and presentation was perfect.",
    "Fresh ingredients and tasty meals.",
    "Loved the ambience and the food was top-notch.",
    "Service was fast and food was cooked to perfection.",
    "Great place for family dinners and celebrations.",
] * 12  # 10 kalimat x 12 = 120

negative_reviews = [
    "Terrible food, really disappointed.",
    "The service was slow and the waiter was rude.",
    "I will never come back to this place.",
    "Food was cold and tasteless.",
    "The place was dirty and not well-maintained.",
    "Overpriced and underwhelming meals.",
    "Had to wait too long for our food.",
    "The quality of ingredients was poor.",
    "Not worth the money or time.",
    "Disappointed with the overall experience.",
] * 12  # 10 kalimat x 12 = 120

reviews = positive_reviews + negative_reviews
sentiments = ['Positive'] * len(positive_reviews) + ['Negative'] * len(negative_reviews)

df = pd.DataFrame({
    'review': reviews,
    'sentiment': sentiments
})

print("Jumlah data:", len(df))

# Preprocess review text
df['clean_review'] = df['review'].apply(preprocess_text)

# Tambah kolom panjang review
df['review_length'] = df['clean_review'].apply(lambda x: len(x.split()))

# Label encoding
df['label'] = df['sentiment'].map({'Positive': 1, 'Negative': 0})

# Split data
X = df['clean_review']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizer & training model
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

# Evaluasi
print(f"Akurasi: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Visualisasi WordCloud positif dan negatif
pos_text = ' '.join(df[df['label'] == 1]['clean_review'])
neg_text = ' '.join(df[df['label'] == 0]['clean_review'])

wc_pos = WordCloud(width=400, height=300, background_color='white').generate(pos_text)
wc_neg = WordCloud(width=400, height=300, background_color='white').generate(neg_text)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(wc_pos, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Positive Reviews')

plt.subplot(1,2,2)
plt.imshow(wc_neg, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Negative Reviews')
plt.show()

# Visualisasi distribusi panjang review
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='review_length', hue='sentiment', bins=15, kde=True, palette=['red', 'green'])
plt.title('Distribusi Panjang Review per Sentimen')
plt.xlabel('Jumlah Kata per Review')
plt.show()

# Visualisasi top kata di masing-masing kelas (dengan CountVectorizer)
cv = CountVectorizer(stop_words='english', max_features=10)
X_cv = cv.fit_transform(df['clean_review'])
feature_names = cv.get_feature_names_out()

df_words = pd.DataFrame(X_cv.toarray(), columns=feature_names)
df_words['label'] = df['label']

# Hitung frekuensi rata-rata tiap kata per label
freq_pos = df_words[df_words['label'] == 1].mean().sort_values(ascending=False)
freq_neg = df_words[df_words['label'] == 0].mean().sort_values(ascending=False)

fig, axes = plt.subplots(1,2, figsize=(14,5))
freq_pos.plot.bar(ax=axes[0], color='green')
axes[0].set_title('Top 10 Kata di Review Positif')
axes[0].set_ylabel('Rata-rata Frekuensi')

freq_neg.plot.bar(ax=axes[1], color='red')
axes[1].set_title('Top 10 Kata di Review Negatif')

plt.tight_layout()
plt.show()
