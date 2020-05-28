# fastML

--------
A Python package built on sklearn for running a series of classification Algorithms on a given data in a faster and easier way.
# Algorithms

------------
- ### Logistic Regression
- ### Support Vector Machine
- ### Decision Tree Classifier
- ### Random Forest Classifier
- ### K-Nearest Neighbors
- ### NeuralNet Classifier
--------------------------
# Getting started

-----------------

## Install the package
to install the package, navigate to the file directory and run:
```bash
pip install fastML
```
Navigate to folder and install requirements:
```bash
pip install -r requirements.txt
```
## Usage
Assign the variables X and Y to the desired columns and assign the variable size to the desired test_size.  
```python
X = < df.features >
Y = < df.target >
size = < test_size >
```
## Ecoding Categorical Data 
Encode target variable if non-numerical:
```python
from fastML import EncodeCategorical
Y = EncodeCategorical(Y)
```
## Running fastML
```python
fastML(X, Y, size, RandonForestClassifier(), DecisionTreeClassifier(), KNeighborsClassifier(), SVC(),
        include_special_classifier = True, # to include the neural net classifier
        special_classifier_epochs=200,
        special_classifier_nature ='fixed'
)
```
You may also check the test.py file to see the use case.

## Example output
```python
Using TensorFlow backend.

    
   __          _   __  __ _      
  / _|        | | |  \/  | |     
 | |_ __ _ ___| |_| \  / | |        
 |  _/ _` / __| __| |\/| | |     
 | || (_| \__ \ |_| |  | | |____ 
 |_| \__,_|___/\__|_|  |_|______|
                                 
                                 

____________________________________________________
____________________________________________________
Accuracy Score for SVC is 
0.9811320754716981


Confusion Matrix for SVC is 
[[16  0  0]
 [ 0 20  1]
 [ 0  0 16]]


Classification Report for SVC is 
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        16
           1       1.00      0.95      0.98        21
           2       0.94      1.00      0.97        16

    accuracy                           0.98        53
   macro avg       0.98      0.98      0.98        53
weighted avg       0.98      0.98      0.98        53



____________________________________________________
____________________________________________________
____________________________________________________
____________________________________________________
Accuracy Score for RandomForestClassifier is 
0.9622641509433962


Confusion Matrix for RandomForestClassifier is 
[[16  0  0]
 [ 0 20  1]
 [ 0  1 15]]


Classification Report for RandomForestClassifier is 
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        16
           1       0.95      0.95      0.95        21
           2       0.94      0.94      0.94        16

    accuracy                           0.96        53
   macro avg       0.96      0.96      0.96        53
weighted avg       0.96      0.96      0.96        53



____________________________________________________
____________________________________________________
____________________________________________________
____________________________________________________
Accuracy Score for DecisionTreeClassifier is 
0.9622641509433962


Confusion Matrix for DecisionTreeClassifier is 
[[16  0  0]
 [ 0 20  1]
 [ 0  1 15]]


Classification Report for DecisionTreeClassifier is 
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        16
           1       0.95      0.95      0.95        21
           2       0.94      0.94      0.94        16

    accuracy                           0.96        53
   macro avg       0.96      0.96      0.96        53
weighted avg       0.96      0.96      0.96        53



____________________________________________________
____________________________________________________
____________________________________________________
____________________________________________________
Accuracy Score for KNeighborsClassifier is 
0.9811320754716981


Confusion Matrix for KNeighborsClassifier is 
[[16  0  0]
 [ 0 20  1]
 [ 0  0 16]]


Classification Report for KNeighborsClassifier is 
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        16
           1       1.00      0.95      0.98        21
           2       0.94      1.00      0.97        16

    accuracy                           0.98        53
   macro avg       0.98      0.98      0.98        53
weighted avg       0.98      0.98      0.98        53



____________________________________________________
____________________________________________________
____________________________________________________
____________________________________________________
Accuracy Score for LogisticRegression is 
0.9811320754716981


Confusion Matrix for LogisticRegression is 
[[16  0  0]
 [ 0 20  1]
 [ 0  0 16]]


Classification Report for LogisticRegression is 
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        16
           1       1.00      0.95      0.98        21
           2       0.94      1.00      0.97        16

    accuracy                           0.98        53
   macro avg       0.98      0.98      0.98        53
weighted avg       0.98      0.98      0.98        53



____________________________________________________
____________________________________________________
Included special classifier with fixed nature
2020-05-28 15:25:28.257871: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
2020-05-28 15:25:28.257902: E tensorflow/stream_executor/cuda/cuda_driver.cc:313] failed call to cuInit: UNKNOWN ERROR (303)
2020-05-28 15:25:28.257959: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (lyrax): /proc/driver/nvidia/version does not exist
2020-05-28 15:25:28.258194: I tensorflow/core/platform/cpu_feature_guard.cc:143] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-28 15:25:28.284576: I tensorflow/core/platform/profile_utils/cpu_utils.cc:102] CPU Frequency: 2400000000 Hz
2020-05-28 15:25:28.285480: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7fe8b4000b60 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-28 15:25:28.285557: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 4)                 20        
_________________________________________________________________
dense_2 (Dense)              (None, 16)                80        
_________________________________________________________________
dense_3 (Dense)              (None, 3)                 51        
=================================================================
Total params: 151
Trainable params: 151
Non-trainable params: 0
_________________________________________________________________
Train on 97 samples, validate on 53 samples
Epoch 1/200
97/97 [==============================] - 0s 1ms/step - loss: 1.0995 - accuracy: 0.1443 - val_loss: 1.1011 - val_accuracy: 0.3019
97/97 [==============================] - 0s 63us/step - loss: 0.5166 - accuracy: 0.7010 - val_loss: 0.5706 - val_accuracy: 0.6038
Epoch 100/200
97/97 [==============================] - 0s 88us/step - loss: 0.5128 - accuracy: 0.7010 - val_loss: 0.5675 - val_accuracy: 0.6038
Epoch 200/200
97/97 [==============================] - 0s 79us/step - loss: 0.3375 - accuracy: 0.8969 - val_loss: 0.3619 - val_accuracy: 0.9057
97/97 [==============================] - 0s 36us/step
____________________________________________________
____________________________________________________
Accuracy Score for neuralnet is 
0.8969072103500366


Confusion Matrix for neuralnet is 
[[16  0  0]
 [ 0 16  5]
 [ 0  0 16]]


Classification Report for neuralnet is 
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        16
           1       1.00      0.76      0.86        21
           2       0.76      1.00      0.86        16

    accuracy                           0.91        53
   macro avg       0.92      0.92      0.91        53
weighted avg       0.93      0.91      0.91        53



____________________________________________________
____________________________________________________
                    Model            Accuracy
0                     SVC  0.9811320754716981
1  RandomForestClassifier  0.9622641509433962
2  DecisionTreeClassifier  0.9622641509433962
3    KNeighborsClassifier  0.9811320754716981
4      LogisticRegression  0.9811320754716981
5               neuralnet  0.8969072103500366

```
## Author: [Jerry Buaba](linkedin.com/in/jerry-buaba-768351172)
## Acknowledgements
Thanks to [Vincent Njonge](linkedin.com/in/vincent-njonge-528070178), [Philemon Johnson](linkedin.com/in/philemon-johnson-b95009171), [William Akuffo](linkedin.com/in/william-akuffo-26b430159), [Labaran Mohammed](linkedin.com/in/adam-labaran-111358181), [Benjamin Acquaah](linkedin.com/in/benjamin-acquaah-9294aa14b), [Silas Bempong](https://www.linkedin.com/in/silas-bempong-604916120) and [Emmanuel Amouku](linkedin.com/in/emmanuel-amoaku) for making this project a success.

