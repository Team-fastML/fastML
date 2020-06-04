from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils

class neuralnet:

    def __init__(self,xtrain = None,xtest = None,ytrain = None,ytest = None,nature = 'fixed',verbose = 1):
        '''
        NB:
        -- encode_categorically(self,Y) :needed to categrically encode labels for the right output shape of the neural net
        -- normalize(self,X) : is required to normalize features within a set feature range. The default is from 0 to 1
        -- set verbose : to 0 or 1 or 2. Determins weather outpup training info should be printed. 0 means hold the output
        
        under dev (in future commits)
        -- nature : detremines whether the neural archetecture should be fixed or adaptive to the dataset
                    fixed will use the default architecture
                    adaptive will adapt the architecture to the size and shape of the dataset to include more layers and
                    trainable parameters in hopes to imporve accuracy *include this in future commits
        --- training visualization
        
        



        
        
        '''
        self.xtrain = self.normalize(xtrain)
        self.ytrain = self.encode_categorically(ytrain)
        self.xtest = self.normalize(xtest)
        self.nature = nature
        self.ytest = self.encode_categorically(ytest)
        self.verbose = verbose


    def set_model(self):
        #fixed neural architecture
        model = Sequential()
        model.add(Dense(units =self.xtrain.shape[1],input_dim = self.xtrain.shape[1],activation = 'relu'))
        model.add(Dense(units =16,activation = 'relu'))
        model.add(Dense(units =self.ytrain.shape[1],activation = 'softmax'))
        model.summary()
        model.compile(optimizer = 'Adam',loss ='categorical_crossentropy',metrics = ['accuracy'])

        return model

    def normalize(self,X):
        sc = MinMaxScaler(feature_range=(0,1))
        x_train = sc.fit_transform(X)
        return x_train

    def encode_categorically(self,Y):
        #in future maybe include type checker for Y for user convinience and robustness
        y_encoded = np_utils.to_categorical(Y)
        return y_encoded

    def fit_neural(self,model = None,epochs = 1):
        #model = self.set_model()
        model.fit(self.xtrain,self.ytrain,verbose = self.verbose,epochs =epochs,validation_data= (self.xtest,self.ytest))
        #in future, may include training history visualization

        #evaluate model on testing data
        performance  = model.evaluate(self.xtrain,self.ytrain) 
        return performance[1] #accuracy

