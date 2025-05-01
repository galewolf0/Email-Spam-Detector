#importing required libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.linear_model import LogisticRegression
import pickle

#loding the pickle file
with open("model","rb") as f:
    model = pickle.load(f)

#predicting if new email is spam or not
newEmail = ["Please come to office tomorrow"]
df = pd.read_csv("dataset/combined_data.csv")
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(df.text)
new_email_vectorized = vectorizer.transform(newEmail)
prediction = model.predict(new_email_vectorized)

if prediction[0] == 1:
    print("Spam")
else:
    print("Ham")