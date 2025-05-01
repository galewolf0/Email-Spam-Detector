#importing required libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

#importing dataset
df = pd.read_csv("dataset/combined_data.csv")

#convert text data into numerical features using Count Vectorization
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(df.text)

#split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, df.label, test_size = 0.2)

#create a multinomial Naive Bayes Classifier
model = LogisticRegression(solver='lbfgs', max_iter=1000)

#training the model
model.fit(x_train, y_train)

#making prediction on test data
y_pred = model.predict(x_test)

#evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy*100}/%")
print("Classification Report:\n", report)

#saving the model
with open("model","wb") as f:
    pickle.dump(model, f)
    print("model saved!!")
    f.close()