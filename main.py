import pandas as pd
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

def zad1():
    X_train = pd.read_csv('./Train/X_train.txt', sep="\t", header=None)
    X_test = pd.read_csv('./Test/X_test.txt', sep="\t", header=None)
#how to add here the labels...
    print('Train data: ')
    print(X_train)
    print('Test data: ')
    print (X_test)


def svm_(): #underscore to not override built in function
    X = [[0, 0], [1, 1]]
    y = [0, 1]
    clf = svm.SVC()
    clf.fit(X, y)
    predicted = clf.predict([[2., 2.]])
    print('SVM: ')
    print(predicted)


def knn():
    X = [[0, 0], [1, 1]]
    y = [0, 1]
    # model = KNeighborsClassifier(n_neighbors=3)
    # based on: "ValueError: Expected n_neighbors <= n_samples,  but n_samples = 2, n_neighbors = 3"
    # I know n_neighbors <= n_samples
    model = KNeighborsClassifier(n_neighbors=2)

    # Train the model using the training sets
    model.fit(X, y)
    # Predict Output
    predicted = model.predict([[0, 2]])  # 0:Overcast, 2:Mild
    print('KNN: ')
    print(predicted)

def decisionTree():
    X = [[0, 0], [1, 1]]
    y = [0, 1]
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()
    # Train Decision Tree Classifer
    clf = clf.fit(X, y)
    predicted = clf.predict(X)
    print('Decision Tree: ')
    print(predicted)

def randomForest():
    X = [[0, 0], [1, 1]]
    y = [0, 1]
    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=100)
    # Train the model using the training sets
    clf.fit(X, y)
    predicted = clf.predict(X)
    print('Random Forest: ')
    print(predicted)


def zad2():
    svm_()
    knn()
    decisionTree()
    randomForest()


zad2()

#IMPORTANT NOTES:
# - IN ZAD1 READ MAYBE SEPARATELY X AND y
# - TO 'FIT' FUNCTIONS USE TRAIN DATA, TO PREDICT - TEST DATA