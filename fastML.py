##importing the needed libraries 

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd

##creating a function to handle classification algorithms

def fastML(X, Y, size): #defining arguments to be passed in function

    ##assigning various classification algoritms to variables for easy accessibilty
    svc = SVC()
    KNN = KNeighborsClassifier()
    DTC = DecisionTreeClassifier()
    RF = RandomForestClassifier()

    ##splitting the data into training and testing data and setting the test_size
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=size)

    ##training the models with the train data
    svc.fit(X_train, y_train)
    KNN.fit(X_train, y_train)
    DTC.fit(X_train, y_train)
    RF.fit(X_train, y_train)

    ##making model predictions on the test data
    DTC_prediction = DTC.predict(X_test)
    KNN_prediction = KNN.predict(X_test)
    SVC_prediction = svc.predict(X_test)
    RF_prediction = RF.predict(X_test)

    ##outputing statistics on the performance of all individual model
    ##the statistics to be printed are the accuracy score, confusion matrix and classification report
    print(
        "Accuracy score for Decision tree Classifier is " + str(accuracy_score(y_test, DTC_prediction)))
    print(
        "Confusion Matrix for Decision tree Classifier is " + str(confusion_matrix(y_test, DTC_prediction)))
    print(
        "Classification Report for Decision tree Classifier is " + str(classification_report(y_test, DTC_prediction)))

    print(
        "Accuracy score for K-Nearest Neighbors is " + str(accuracy_score(y_test, KNN_prediction)))
    print(
        "Confusion Matrix for K-Nearest Neighbors is " + str(confusion_matrix(y_test, KNN_prediction)))
    print(
        "Classification Report for K-Nearest Neighbors is " + str(classification_report(y_test, KNN_prediction)))

    print(
        "Accuracy score for Support Vector Machine is " + str(accuracy_score(y_test, SVC_prediction)))
    print(
        "Confusion Matrix for Support Vector Machine is " + str(confusion_matrix(y_test, SVC_prediction)))
    print(
        "Classification Report for Support Vector Machine is " + str(classification_report(y_test, SVC_prediction)))

    print(
        "Accuracy score for Random Forest Classifier is " + str(accuracy_score(y_test, RF_prediction)))
    print(
        "Confusion Matrix for Random Forest Classifier " + str(confusion_matrix(y_test, RF_prediction)))
    print(
        "Classification Report for Random Forest Classifier is " + str(classification_report(y_test, RF_prediction)))

    ##saving the accuracy scores of individual models as variables
    DTC_accuracy = str(accuracy_score(y_test, DTC_prediction))
    KNN_accuracy = str(accuracy_score(y_test, KNN_prediction))
    SVC_accuracy = str(accuracy_score(y_test, SVC_prediction))
    RF_accuracy = str(accuracy_score(y_test, RF_prediction))

    ##making a list of individual models and their accuracy scores
    acc_score = [['Decision tree', DTC_accuracy], ['K-Nearest Neighbors', KNN_accuracy], ['Support Vector Machine',
                                                                                          SVC_accuracy], ['Random Forest', RF_accuracy]]
    
    ##creating a dataframe of every individual model and accuracy score 
    df = pd.DataFrame(acc_score, columns=['Model', 'Accuracy'])
    ##outputing the created dataframe 
    print(df)

##creating another function for handling the categorial encoding of data 
def EncodeCategorical(Y): ##defining the argument the function takes
    from sklearn.preprocessing import LabelEncoder ##importing the needed library to handle data encoding
    le = LabelEncoder() ## assigning the encoder function to a variable
    return le.fit_transform(Y) ##returning the transformed encoded data
