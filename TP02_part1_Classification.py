# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 12:34:29 2022

@author: Pix Info
"""



# **Classification Algorithms**

# **Data Preprocessing**


# Import Python Libraries
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("/content/breast_cancer.csv")

# Display dimensions of dataframe
print(df.shape)
print(df.info())

print("-----------------------------------------------------------------------")
# Display first 10 records
print(df.head(10))

print("-----------------------------------------------------------------------")
# List the column names
print(df.columns)


print("-----------------------------------------------------------------------")
# Display statistics for numeric columns
print(df.describe())



"""# **Correlation and Data Visualization**"""

# Find the pairwise correlation of all columns in the dataframe.
print(df.corr())

# Heatmap
print("\n")
corrMatrix = df.corr()
fig, ax = plt.subplots(figsize=(16,16)) 
#sns.heatmap(corrMatrix, annot=True)
sns.heatmap(corrMatrix, annot=True, linewidth=0.01, square=True, cmap="RdBu", linecolor="black")

bins = 5

plt.figure(figsize=(15,15))
plt.subplot(2, 2, 1)
sns.distplot(df[df['diagnosis']=='M']['radius_mean'], bins=bins, color='green', label='M')
sns.distplot(df[df['diagnosis']=='B']['radius_mean'], bins=bins, color='red', label='B')
plt.legend(loc='upper right')
plt.subplot(2, 2, 2)
sns.distplot(df[df['diagnosis']=='M']['texture_mean'], bins=bins, color='green', label='M')
sns.distplot(df[df['diagnosis']=='B']['texture_mean'], bins=bins, color='red', label='B')
plt.legend(loc='upper right')

plt.subplot(2, 2, 3)
sns.distplot(df[df['diagnosis']=='M']['perimeter_mean'], bins=bins, color='green', label='M')
sns.distplot(df[df['diagnosis']=='B']['perimeter_mean'], bins=bins, color='red', label='B')
plt.legend(loc='upper right')
plt.subplot(2, 2, 4)
sns.distplot(df[df['diagnosis']=='M']['area_mean'], bins=bins, color='green', label='M')
sns.distplot(df[df['diagnosis']=='B']['area_mean'], bins=bins, color='red', label='B')
plt.legend(loc='upper right')

plt.tight_layout()

plt.show()



X = my_data.drop(["diagnosis"],axis=1).values
y = my_data["diagnosis"].values
print("X : ",X.shape," y : ",y.shape)

# Split into Input and Output Elements
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, 
              y, test_size= 0.20, random_state=100, stratify=y)

print("X_train = ",X_train.shape ," y_train = ", y_train.shape)
print("X_test  = ",X_test.shape ," y_test = ", y_test.shape)





"""# **Support Vector Machines SVM**"""

from sklearn import datasets, svm, metrics

# Create a classifier: a support vector classifier
# classifier = svm.SVC()
classifier = svm.SVC(kernel="linear")
# classifier = svm.SVC(kernel="poly", degree=5, C=10, gamma=0.01)
# classifier = svm.SVC(kernel="sigmoid", C=10, gamma=0.001)
# classifier = svm.SVC(kernel="rbf", C=10, gamma=0.01)

# Train the classifier
classifier.fit(X_train, y_train)

# Now predict the value of X_test
predicted = classifier.predict(X_test)

# Classification report
print("Classification report : \n", classifier,"\n", 
      metrics.classification_report(y_test, predicted))

disp = metrics.plot_confusion_matrix(classifier, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")

print("Confusion matrix: \n", disp.confusion_matrix)





"""# **Logistic Regression Classifier**

"""

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Create the classifier
# multi_class: default = "auto"
logreg = LogisticRegression(random_state=42)

# Train the classifier
logreg.fit(X_train,y_train)

# Predict the value of X_test
predicted = logreg.predict(X_test)

# Classification report
print("Classifier : ", logreg)
print("Classification report for classifier : \n", metrics.classification_report(y_test, predicted))

disp = metrics.plot_confusion_matrix(logreg, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix: \n", disp.confusion_matrix)



"""# **Stochastic Gradient Descent**"""

from sklearn.linear_model import SGDClassifier
from sklearn import metrics

# Create the classifier
sgdc_cls = SGDClassifier(loss="log", alpha=0.01, penalty="l2", random_state =42)

# Train the classifier
sgdc_cls.fit(X_train,y_train)

# Predict the value of X_test
predicted = sgdc_cls.predict(X_test)

# Classification report
print("Classifier : ", sgdc_cls)
print("Classification report for classifier : \n", metrics.classification_report(y_test, predicted))

disp = metrics.plot_confusion_matrix(sgdc_cls, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix: \n", disp.confusion_matrix)


"""# **Decision Trees**"""

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# training the model on training set
# criterion='gini' splitter='best' 
tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train, y_train)

# making predictions on the testing set 
predicted = tree_clf.predict(X_test)

# comparing actual response values (y_test) with predicted response values (predicted)
from sklearn import metrics 
print("Classification report : \n", tree_clf,"\n", metrics.classification_report(y_test, predicted))
disp = metrics.plot_confusion_matrix(tree_clf, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix: \n", disp.confusion_matrix)


