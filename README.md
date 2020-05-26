# fastML
A Python package built on sklearn for running a series of classification Algorithms on a given data in a faster and easier way.
# Algorithms
## Support Vector Machine
## Decision Tree Classifier
## Random Forest Classifier
## K-Nearest Neighbors

# Getting started
fork and clone this repo on your local machine and set-up your virtualenv by running the command 
```bash
pip install -r requirements.txt
```
# Installing the package
to install the package, navigate to the file directory and run:
```bash
pip install .
```
## Usage
Assign the variables X and Y to the desired columns and assign the variable size to the desired test_size.
## Ecoding Categorical Data 
```python 
from fastML import EncodeCategorical
Y = EncodeCategorical(Y)
```

Check test.py to see the use case.
```python
from fastML import fastML
size = float(input('Enter Value for test_size:'))
fastML(X, Y, size, RandonForestClassifier(), DecisionTreeClassifier(), KNeighborsClassifier(), SVC())
```
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
