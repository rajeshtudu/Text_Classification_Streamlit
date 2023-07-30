import numpy as np
import pandas as pd
import re,string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
import streamlit as st

df = pd.read_csv('cleaned_news.csv', encoding = 'latin1')
df = df.sample(frac = 1)

from sklearn.linear_model import LogisticRegression
log_regression = LogisticRegression()

vectorizer = TfidfVectorizer(stop_words="english")

X = df['cleaned']
Y = df['category']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15) #Splitting dataset

# #Creating Pipeline
pipeline = Pipeline([('vect', vectorizer),
                     ('chi',  SelectKBest(chi2, k=1450)),
                     ('clf', LogisticRegression(random_state=0))])

model = pipeline.fit(X_train, y_train)

# file = open('news.txt','r')
# news = file.read()
# file.close()

# news = input("Enter news = ")
def predict_news(txt):
    news_data = {'predict_news':[txt]}
    news_data_df = pd.DataFrame(news_data)
    predict_news_cat = model.predict(news_data_df['predict_news'])[0]
    return predict_news_cat


st.title("Text Classification")
news = st.text_area('Enter news')
if st.button('Submit'):
    st.write("Predicted news category = ", predict_news(news))