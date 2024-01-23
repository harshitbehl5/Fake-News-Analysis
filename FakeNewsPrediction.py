import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk 

# loading the dataset to a pandas DataFrame
news_dataset = pd.read_csv("D:/Data Analytics/DA Python/Datasets/train.csv")
# replacing the null values with empty string
news_dataset = news_dataset.fillna('')
# merging the author name and news title
news_dataset['content'] = news_dataset['author']+' '+news_dataset['title']

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)

train_articles, test_articles, train_labels, test_labels = train_test_split(news_dataset['content'].values, news_dataset['label'].values, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer()
train_features = tfidf.fit_transform(train_articles)
test_features = tfidf.transform(test_articles)

model1 = LogisticRegression()
model1.fit(train_features, train_labels)

new_articles = "JulianRoberts Global temperatures are rising"
new_articles = stemming(new_articles)
new_features = tfidf.transform([new_articles])
new_predicted_labels = model1.predict(new_features)


# X_test_prediction = model1.predict(test_features)
# test_data_accuracy = accuracy_score(X_test_prediction, test_labels)
# print(test_data_accuracy)