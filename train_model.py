#ipynb export to py
# %%


# %%
import pandas as pd

df = pd.read_csv(r'E:/ml projects/Fake_Real_Data.csv')
display(df.head())


# %%
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


nltk.download('stopwords')
nltk.download('punkt')


df['Text'] = df['Text'].fillna('')


def preprocess_text(text):

    text = text.lower()

    text = re.sub(r'[^a-z\s]', '', text)

    tokens = text.split()

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)


df['Processed_Text'] = df['Text'].apply(preprocess_text)


display(df.head())

# %%
from sklearn.feature_extraction.text import TfidfVectorizer


tfidf_vectorizer = TfidfVectorizer(max_features=5000)


tfidf_matrix = tfidf_vectorizer.fit_transform(df['Processed_Text'])


print("Shape of TF-IDF matrix:", tfidf_matrix.shape)

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier

X = tfidf_matrix
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pac = PassiveAggressiveClassifier(max_iter=50) 

pac.fit(X_train, y_train)

print("PassiveAggressiveClassifier trained successfully.")

# %%
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

y_train_pred = pac.predict(X_train)

y_test_pred = pac.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)

test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")

cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=pac.classes_, yticklabels=pac.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# %%
import pickle

with open(r'E:/ml projects/model.pkl', 'wb') as model_file:
    pickle.dump(pac, model_file)

with open(r'E:/ml projects/vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully.")


