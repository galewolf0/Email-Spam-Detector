#importing required modules
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.uic import loadUi
import sys
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

#
class Ui_Main(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui_Main, self).__init__()
        loadUi("Main.ui", self)
        
        #hiding elements and making them uneditable
        self.resultMessage.setEnabled(False)
        self.spamMessage.setEnabled(False)
        self.notSpamMessage.setEnabled(False)
        self.spamMessage.hide()
        self.notSpamMessage.hide()

        #button connections
        self.scanButton.clicked.connect(lambda: self.scan())
        self.clearButton.clicked.connect(lambda: self.clear())

    #Processing
    def scan(self):
        #loding the pickle file
        with open("model","rb") as f:
            model = pickle.load(f)

        #predicting if new email is spam or not
        newEmail = [self.toCheck.toPlainText()]
        df = pd.read_csv("dataset/combined_data.csv")
        vectorizer = CountVectorizer()
        x = vectorizer.fit_transform(df.text)
        new_email_vectorized = vectorizer.transform(newEmail)
        prediction = model.predict(new_email_vectorized)

        #Displaying the result
        self.resultMessage.hide()
        self.scanButton.setEnabled(False)
        if prediction[0] == 1:
            self.spamMessage.show()
        else:
            self.notSpamMessage.show()

    #What happens when clear button is clicked
    def clear(self):
        #hiding and unhiding elements
        self.spamMessage.hide()
        self.notSpamMessage.hide()
        self.resultMessage.show()

        #clearing the text box
        self.toCheck.setPlainText("")
        self.scanButton.setEnabled(True)


#main
app = QtWidgets.QApplication(sys.argv)
stack = QtWidgets.QStackedWidget()
Main = Ui_Main()
stack.addWidget(Main)
stack.setCurrentWidget(Main)
stack.show()
sys.exit(app.exec_())