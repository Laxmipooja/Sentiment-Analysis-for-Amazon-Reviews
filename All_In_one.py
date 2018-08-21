import csv
import pandas as p
import string
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
messages = p.read_csv('/Users/laxmipoojaanegundi/Desktop/HW2/reviews.csv',sep='|',names=['labels','text'])
messages=messages.drop([0])
length = len(messages)
df1 = list(range(4,length,5))
messages1 = messages.ix[df1]
messages = messages.drop(df1)

messages = messages.dropna(subset=['text'])
messages1 = messages1.dropna(subset=['text'])

count_vect = CountVectorizer()
count_vect1 = count_vect

X_train_counts = count_vect.fit_transform(messages['text'].values.astype('U'))
Y_train_counts = count_vect1.transform(messages1['text'].values.astype('U'))
print(X_train_counts.shape)
print(Y_train_counts.shape)



#----------------SVM----------------
classifier_rbf = LinearSVC(random_state=0)
classifier_rbf.fit(X_train_counts,messages['labels'])
prediction_rbf = classifier_rbf.predict(Y_train_counts)
print("For the SVM classifier:")
print("Accuracy for SVM")
print(accuracy_score(messages1['labels'],prediction_rbf))

clf = MLPClassifier(alpha=1e-5,random_state=1,hidden_layer_sizes=(1000,),learning_rate='adaptive')
clf.fit(X_train_counts,messages['labels'])
print("For the MLP classifier:")
prediction_rbf=clf.predict(Y_train_counts)
print("For the accuracy of  MLP")
print(accuracy_score(messages1['labels'],prediction_rbf))

#---------------------Random Forest-----------------------------

clf = RandomForestClassifier(random_state=0)
clf.fit(X_train_counts,messages['labels'])
prediction_rbf=clf.predict(Y_train_counts)
print("For the Random classifier:")
print (prediction_rbf)
print("For the accuracy for RandomForest")
print(accuracy_score(messages1['labels'],prediction_rbf))
