import pandas as pd
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time

X_train = None
X_test = None
y_train = None
y_test = None

svmLabels = None
knnLabels = None
dtLabels = None
rfLabels = None


def zad1():
    global X_train, X_test, y_train, y_test
    X_train = pd.read_csv('./Train/X_train.txt', sep=" ", header=None)
    X_test = pd.read_csv('./Test/X_test.txt', sep=" ", header=None)
    y_train = pd.read_csv('./Train/y_train.txt', sep=" ", header=None)
    y_test = pd.read_csv('./Test/y_test.txt', sep=" ", header=None)
    print(y_test)


def svm_():  # underscore to not override built in function
    global X_train, X_test, y_train, y_test
    X = X_train
    y = y_train
    clf = svm.SVC()
    clf.fit(X, y.values.ravel())
    predicted = clf.predict(X_test)
    print('SVM: ')
    print(predicted)
    return predicted


def knn():
    global X_train, X_test, y_train, y_test
    X = X_train
    y = y_train
    model = KNeighborsClassifier(n_neighbors=2)

    # Train the model using the training sets
    model.fit(X, y.values.ravel())
    # Predict Output
    predicted = model.predict(X_test)
    print('KNN: ')
    print(predicted)
    return predicted


def decisionTree():
    global X_train, X_test, y_train, y_test
    X = X_train
    y = y_train
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()
    # Train Decision Tree Classifer
    clf = clf.fit(X, y.values.ravel())
    predicted = clf.predict(X_test)
    print('Decision Tree: ')
    print(predicted)
    return predicted


def randomForest():
    global X_train, X_test, y_train, y_test
    X = X_train
    y = y_train
    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=100)
    # Train the model using the training sets
    clf.fit(X, y.values.ravel())
    predicted = clf.predict(X_test)
    print('Random Forest: ')
    print(predicted)
    return predicted


def zad2():
    global svmLabels, knnLabels, dtLabels, rfLabels
    svmLabels = svm_()
    knnLabels = knn()
    dtLabels = decisionTree()
    rfLabels = randomForest()


def calculateConfusionMatrix(y, y_test):
    cm = confusion_matrix(y, y_test).ravel()
    print("CM:")
    print(cm)

    ## TN, FP, FN, TP podobno robi sie tylko dla danych binarnych???
    # tn, fp, fn, tp = confusion_matrix(y, y_test).ravel()
    # print('Confusion matrix')
    # print((tn, fp, fn, tp))


def calculateACC(y, y_test):
    acc = accuracy_score(y, y_test)
    print("ACC:")
    print(acc)
    return acc  # return dodany na potrzeby zadania 4


def calculateRecall(y, y_test):
    rec = recall_score(y, y_test, average='weighted')
    print("Recall:")
    print(rec)


def calculateF1(y, y_test):
    f1 = f1_score(y, y_test, average='weighted')
    print("F1")
    print(f1)
    return f1  # return dodany na potrzeby zadania 4


def calculateAUC(y, y_test):
    auc = roc_auc_score(y, y_test, multi_class='ovo')
    print("AUC")
    print(auc)


def zad3():
    # confusion matrix
    calculateConfusionMatrix(svmLabels, y_test)
    calculateConfusionMatrix(knnLabels, y_test)
    calculateConfusionMatrix(dtLabels, y_test)
    calculateConfusionMatrix(rfLabels, y_test)

    # ACC
    calculateACC(svmLabels, y_test)
    calculateACC(knnLabels, y_test)
    calculateACC(dtLabels, y_test)
    calculateACC(rfLabels, y_test)

    # Recall
    calculateRecall(svmLabels, y_test)
    calculateRecall(knnLabels, y_test)
    calculateRecall(dtLabels, y_test)
    calculateRecall(rfLabels, y_test)

    # F1
    calculateF1(svmLabels, y_test)
    calculateF1(knnLabels, y_test)
    calculateF1(dtLabels, y_test)
    calculateF1(rfLabels, y_test)

    # AUC niby wymaga normalizacji danych żeby działać
    # calculateAUC(svmLabels, y_test)
    # calculateAUC(knnLabels, y_test)
    # calculateAUC(dtLabels, y_test)
    # calculateAUC(rfLabels, y_test)


def svm_CV():
    global X_train, y_train
    X = X_train
    y = y_train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0)

    X_train.shape, y_train.shape

    X_test.shape, y_test.shape

    clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    clf.score(X_test, y_test)

    clf = svm.SVC()
    clf.fit(X, y.values.ravel())
    predicted = clf.predict(X_test)
    print('SVM CV: ')
    print(predicted)
    return predicted


def knn_CV():
    global X_train, y_train
    X = X_train
    y = y_train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0)

    X_train.shape, y_train.shape

    X_test.shape, y_test.shape

    model = KNeighborsClassifier(n_neighbors=2)

    # Train the model using the training sets
    model.fit(X, y.values.ravel())
    # Predict Output
    predicted = model.predict(X_test)
    print('KNN: ')
    print(predicted)
    return predicted


def decisionTree_CV():
    global X_train, y_train
    X = X_train
    y = y_train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0)

    X_train.shape, y_train.shape

    X_test.shape, y_test.shape

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()
    # Train Decision Tree Classifer
    clf = clf.fit(X, y.values.ravel())
    predicted = clf.predict(X_test)
    print('Decision Tree: ')
    print(predicted)
    return predicted


def randomForest_CV():
    global X_train, y_train
    X = X_train
    y = y_train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0)

    X_train.shape, y_train.shape

    X_test.shape, y_test.shape

    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=100)
    # Train the model using the training sets
    clf.fit(X, y.values.ravel())
    predicted = clf.predict(X_test)
    print('Random Forest: ')
    print(predicted)
    return predicted


def zad3_2():
    # Cross Validation is a technique which involves reserving a particular sample of a dataset
    # on which you do not train the model.
    # Later, you test your model on this sample before finalizing it.
    svm_CV()
    knn_CV()
    decisionTree_CV()
    randomForest_CV()


def zad4():
    global X_train, X_test, y_train, y_test

    # Na podstawie wskaźników z zadania 3, dla domyślnych parametrów najlepiej sobie radził SVM, więc tego będę się trzymał
    # gdybym miał testować kombinacje wszystkich parametrów to program wykonywałby się 3 dni
    # dlatego będę manipulował 6 wybranymi parametrami, dla każdego z nich kilka wariantów
    # co samo w sobie daje dużo przebiegów samego algorytmu uczącego + sprawdzenie jego wskaźników
    C = [1, 3]
    degree = [1, 5]
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    gamma = ['auto', 'scale']
    coef0 = [0, 2]
    decision_function_shape = ['ovo', 'ovr']

    bestA = bestB = bestC = bestD = bestE = bestF = None  # 'optymalne' parametry
    bestACC = bestF1 = 0  # sprawdzam tylko ACC i F1 dla zaoszczędzenia czasu
    loopCounter = 1  # licznik żebym wiedział co się dzieje
    # przy powyższym zestawie parametrów uczenie wykona się 2*2*4*2*2*2 = 128 razy
    startTime = time.time()
    for a in C:
        for b in degree:
            for c in kernel:
                for d in gamma:
                    for e in coef0:
                        for f in decision_function_shape:
                            print("Wykonuję kombinację numer: " + str(loopCounter))
                            clf = svm.SVC(C=a, degree=b, kernel=c, gamma=d, coef0=e, decision_function_shape=f)
                            clf.fit(X_train, y_train.values.ravel())
                            predicted = clf.predict(X_test)
                            acc = calculateACC(predicted, y_test)
                            f1 = calculateF1(predicted, y_test)
                            # print('WSKAZNIKI')
                            # print(acc)
                            # print(f1)
                            aver = (acc + f1) / 2
                            oldAver = (bestACC + bestF1) / 2
                            if (aver > oldAver):
                                print("Jest nowy najlepszy")
                                bestA = a
                                bestB = b
                                bestC = c
                                bestD = d
                                bestE = e
                                bestF = f
                                bestACC = acc
                                bestF1 = f1
                            else:
                                if(aver == oldAver):
                                    print("To samo co najlepszy")
                                else:
                                    print("Wcześniej było lepiej")
                            loopCounter = loopCounter + 1
    stopTime = time.time()
    print("Zakończyłem sprawdzanie: "+str(bestA)+"; "+str(bestB)+"; "+bestC+"; "+bestD+"; "+str(bestE)+"; "+bestF)
    print("Wykonywalem się: "+str(((stopTime-startTime) / 60))+" minut")
    return [bestA, bestB, bestC, bestD, bestE, bestF]


zad1()
zad2()
zad3()
zad3_2()
###### Zad4 - nie uruchamiać bez potrzeby
###### przy bieżących ustawieniach na mojej maszynie funkcja wykonywała się 53 minuty
###### i zwróciła wynik [1, 5, poly, scale, 2, ovo]
# zad4()

