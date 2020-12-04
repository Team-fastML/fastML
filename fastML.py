
def fastML(X, Y, size):
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn import utils
    import pandas as pd
    from xgboost import XGBClassifier


    SVC = SVC()
    KNN = KNeighborsClassifier()
    DTC = DecisionTreeClassifier()
    RF = RandomForestClassifier()
    ABC = AdaBoostClassifier()
    XGB = XGBClassifier()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=size)

    SVC.fit(X_train, y_train)
    KNN.fit(X_train, y_train)
    DTC.fit(X_train, y_train)
    RF.fit(X_train, y_train)
    ABC.fit(X_train, y_train)
    XGB.fit(X_train, y_train)

    DTC_prediction = DTC.predict(X_test)
    KNN_prediction = (KNN.predict(X_test))
    SVC_prediction = SVC.predict(X_test)
    RF_prediction = RF.predict(X_test)
    ABC_prediction = ABC.predict(X_test)
    XGB_prediction = XGB.predict(X_test)


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

    print(
        "Accuracy score for AdaBoost Classifier is " + str(accuracy_score(y_test, ABC_prediction)))
    print(
        "Confusion Matrix for AdaBoost Classifier " + str(confusion_matrix(y_test, ABC_prediction)))
    print(
        "Classification Report for AdaBoost Classifier is " + str(classification_report(y_test, ABC_prediction)))

    print(
        "Accuracy score for Extreme Gradient Boosting(XGB) Classifier is " + str(accuracy_score(y_test, XGB_prediction)))
    print(
        "Confusion Matrix for Extreme Gradient Boosting(XGB) Classifier " + str(confusion_matrix(y_test, XGB_prediction)))
    print(
        "Classification Report for Extreme Gradient Boosting(XGB) Classifier is " + str(classification_report(y_test, XGB_prediction)))
    DTC_accuracy = str(accuracy_score(y_test, DTC_prediction))
    KNN_accuracy = str(accuracy_score(y_test, KNN_prediction))
    SVC_accuracy = str(accuracy_score(y_test, SVC_prediction))
    RF_accuracy = str(accuracy_score(y_test, RF_prediction))
    ABC_accuracy = str(accuracy_score(y_test, ABC_prediction))
    XGB_accuracy = str(accuracy_score(y_test, XGB_prediction))

    acc_score = [['Decision tree', DTC_accuracy], ['K-Nearest Neighbors', KNN_accuracy], ['Support Vector Machine',
                                                                                          SVC_accuracy], ['Random Forest', RF_accuracy], ['AdaBoost', ABC_accuracy], ['XGBoost', XGB_accuracy]]
    df = pd.DataFrame(acc_score, columns=['Model', 'Accuracy'])
    print(df)




