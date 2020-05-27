##importing the needed libraries 

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy
##creating a function to handle classification algorithms


def fastML(X, Y, size, *args, special_classifier_epochs=1,special_classifier_nature ='fixed',
          include_special_classifier = False,special_classifier_verbose= 1): #defining arguments to be passed in function
    print('''
    
   __          _   __  __ _      
  / _|        | | |  \/  | |     
 | |_ __ _ ___| |_| \  / | |        
 |  _/ _` / __| __| |\/| | |     
 | || (_| \__ \ |_| |  | | |____ 
 |_| \__,_|___/\__|_|  |_|______|
                                 
                                 
''')
   ## dropping rows containing missing values

    X = pd.DataFrame(X)
    X.dropna(inplace= True, axis = 0)

    Y = pd.DataFrame(Y)
    Y.dropna(inplace= True, axis = 0)
    
    ##splitting the data into training and testing data and setting the test_size
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=size, random_state = 0)   

    ##training the model with train data
    acc_score = []
    for alg in args:
        alg.fit(X_train, y_train.values.ravel())
        prediction = alg.predict(X_test)

        ##outputing statistics on the performance of the individual models
        ##statistics printed out are the accuracy score, confusion matrix and clasification report
        print('____________________________________________________')
        print('____________________________________________________')
        print("Accuracy Score for "+ alg.__class__.__name__+" is \n"+ str(accuracy_score(y_test, prediction)))
        print('\n')

        print("Confusion Matrix for "+ alg.__class__.__name__+" is \n"+ str(confusion_matrix(y_test, prediction)))
        print('\n')

        print("Classification Report for "+ alg.__class__.__name__+" is \n"+ str(classification_report(y_test, prediction)))
        print('\n')

        print('____________________________________________________')
        print('____________________________________________________')

        ##saving the accuracy scores of the individual models as variables
        accuracy = str(accuracy_score(y_test, prediction))

        ##making a list of individual models and their accuracy scores
        acc_score.append([alg.__class__.__name__, accuracy])
    
    if include_special_classifier:
        print('Included special classifier with',special_classifier_nature,'nature')
        from specialClassificationModel import neuralnet
        NN = neuralnet(xtrain = X_train,xtest= X_test,ytrain=y_train,ytest = y_test,
                        nature = special_classifier_nature,verbose = special_classifier_verbose) 
        neural_model = NN.set_model()
        neural_model_test_acc = NN.fit_neural(model = neural_model,epochs = special_classifier_epochs )
        acc_score.append([NN.__class__.__name__,str(neural_model_test_acc)])


        
        prediction = neural_model.predict_classes(NN.normalize(X_test))
        

        print('____________________________________________________')
        print('____________________________________________________')
        print("Accuracy Score for "+ NN.__class__.__name__+" is \n"+ str(neural_model_test_acc))
        print('\n')

        print("Confusion Matrix for "+ NN.__class__.__name__+" is \n"+ str(confusion_matrix(y_test,prediction)))
        print('\n')

        print("Classification Report for "+ NN.__class__.__name__+" is \n"+ str(classification_report(y_test, prediction)))
        print('\n')

        print('____________________________________________________')
        print('____________________________________________________')
        


    

    
    ##creating a dataframe of every individual model and accuracy score 
    df = pd.DataFrame(acc_score, columns=['Model', 'Accuracy'])
    ##outputing the created dataframe 
    print(df)

##creating another function for handling the categorial encoding of data 
def EncodeCategorical(Y): ##defining the argument the function takes
    from sklearn.preprocessing import LabelEncoder ##importing the needed library to handle data encoding
    le = LabelEncoder() ## assigning the encoder function to a variable
    return le.fit_transform(Y) ##returning the transformed encoded data 
