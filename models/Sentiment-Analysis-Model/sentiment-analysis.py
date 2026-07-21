import os
import html
import tempfile
import regex as re
import pandas as pd
from joblib import Memory

from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

with tempfile.TemporaryDirectory() as temp_dir:
    api = KaggleApi()
    api.authenticate()
    print("Fetching dataset...")
    api.dataset_download_files("kazanova/sentiment140", path=temp_dir, unzip=True)
    target_file = os.path.join(temp_dir, os.listdir(temp_dir)[0])
    dataset = pd.read_csv(target_file, encoding='latin-1', header=None) 

df = dataset[[0, 5]].copy() 
df.columns = ['sentiment', 'text']

# Exploratory Data Analysis and Data Preprocessing
for text in df['text'].head(5):
    print(text)

print(df.shape)

df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})
print(df['sentiment'].value_counts()) # balanced dataset
print(df.isnull().sum())
print(df['text'].duplicated().sum()) # 18534
print(df['text'].str.len().describe())

def clean_text(text):
    text = html.unescape(text)                          # unescape HTML entities
    text = text.lower()                                 # lowercase
    text = re.sub(r'https?://\S+|www\.\S+', '', text)   # remove URLs
    text = re.sub(r'@\w+', '', text)                    # remove mentions
    text = re.sub(r'#(\w+)', r'\1', text)               # remove hashtags -> keep word
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)          # normalize elongated chars
    text = re.sub(r'[^a-z0-9\s:;\-)(dp]', '', text)     # remove punctuation, keep emoticon chars
    text = re.sub(r'\s+', ' ', text).strip()            # remove extra whitespace
    return text

# outliers
print(df.loc[df['text'].str.len() > 300, 'text'].sample(5, random_state=42))
print(df.loc[df['text'].str.len() < 15, 'text'].sample(5, random_state=42))

df['text'] = df['text'].apply(clean_text)
df = df.drop_duplicates(subset='text', keep='first')

print(df['text'].str.len().quantile([0.90, 0.95, 0.99])) # char length quantiles - 130, 136, 141

x_train, x_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.3, random_state=42)

pipeline_lr = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=20000, sublinear_tf=True, ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=1000, solver='lbfgs', C=2.0, random_state=42))], memory=Memory("./cache", verbose=0))

pipeline_nb = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=20000, sublinear_tf=True, ngram_range=(1, 2))),
    ('clf', MultinomialNB(alpha=1.0))], memory=Memory("./cache", verbose=0))

pipeline_sgd = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=20000, sublinear_tf=True, ngram_range=(1, 2))),
    ('clf', SGDClassifier(max_iter=2000, loss='hinge', alpha=0.0001, random_state=42))], memory=Memory("./cache", verbose=0))

pipeline_lr.fit(x_train, y_train)
pipeline_nb.fit(x_train, y_train)
pipeline_sgd.fit(x_train, y_train)

y_pred_lr = pipeline_lr.predict(x_test)
y_pred_nb = pipeline_nb.predict(x_test)
y_pred_sgd = pipeline_sgd.predict(x_test)

print("Logistic Regression Report:")
print(classification_report(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))
print("Naive Bayes Report:")
print(classification_report(y_test, y_pred_nb))
print(confusion_matrix(y_test, y_pred_nb))
print("Linear SGD Report:")
print(classification_report(y_test, y_pred_sgd))
print(confusion_matrix(y_test, y_pred_sgd))

# best performing model is Logistic Regression based on the classification report and confusion matrix results. 
# precision = 0.80, recall = 0.82, f1-score = 0.81