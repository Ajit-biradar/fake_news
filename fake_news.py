import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split #split data into train and test sets
from sklearn.feature_extraction.text import CountVectorizer #convert text comment into a numeric vector
from sklearn.feature_extraction.text import TfidfTransformer #use TF IDF transformer to change text vector created by count vectorizer
#With Tfidftransformer you will compute word counts using CountVectorizer and then compute the IDF values and only then compute the Tf-idf scores.
#With Tfidfvectorizer you will do all three steps at once.
from sklearn.svm import SVC# Support Vector Machine
from sklearn.pipeline import Pipeline #pipeline to implement steps in series
#from gensim import parsing # To stem data
#Gensim is an open-source library for unsupervised topic modeling and natural language processing, using modern statistical machine learning.
#from joblib import dump, load #save or load model
#from pdftotext import convert_pdf_to_string #converts pdf_to_string
from sklearn.metrics import accuracy_score  #accuracy
import pickle
import numpy as nm
#Read the dataset
df = pd.read_csv("./dataset.csv", encoding='latin1')

#Split the dataset in x and y
#X, y = df['word'], df['taxonomy']
X, y = df['text'], df['deceptive']   

#Split data in train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


#Use pipeline to carry out steps in sequence with a single object
#SVM's rbf kernel gives highest accuracy in this classification problem.
from sklearn.svm import SVC# Support Vector Machine
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SVC(kernel='rbf'))])

#Train the model
text_clf.fit(X_train, y_train)
'''
#print(X_test)
predicted = text_clf.predict(X_test)
acc=accuracy_score(y_test, predicted)
print('Accuracy:',acc)
'''
y_pred= text_clf.predict(X_test)  
x_pred= text_clf.predict(X_train)
#Y_pred=y_pred.astype(int)
print("Printing Y_prediction","\n",y_pred)
print('Train Score: ', text_clf.score(X_train, y_train))  
print('Test Score: ', text_clf.score(X_test, y_test))
#Creating the Confusion matrix  
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as mtp  
cm= confusion_matrix(y_test, y_pred)
print("Printing Confusion matrix","\n",cm)
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test, y_pred)
print("Printing Accuracy","\n",acc)
X_train=nm.arange(0,len(y_train),1)
y_train=nm.arange(0,len(y_train),1)
X_test=nm.arange(0,len(y_train),1)
y_test=nm.arange(0,len(y_train),1)

mtp.scatter(X_train, y_train, color="green")   
mtp.plot(X_train, x_pred, color="red")    
mtp.title("feedback vs status (Training Dataset)")  
mtp.xlabel("feedback")

mtp.ylabel("Status")  
mtp.show()

#visualizing the Test set results  
mtp.scatter(X_test, y_test, color="blue")   
mtp.plot(X_train, x_pred, color="red")    
mtp.title("feedback vs status (Test Dataset)")  
mtp.xlabel("feedback")  
mtp.ylabel("Status")  
mtp.show()
