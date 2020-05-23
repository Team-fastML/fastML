# fastML
A Python package built on sklearn for running a series of classification Algorithms on a given data in a faster and easier way.

# Getting started
fork and clone this repo on your local machine and set-up your virtualenv by running the command 
```bash
pip install requirements.txt
```
# Installing the package
to install the package, navigate to the file directory and run:
```bash
pip install .
```
## Usage
Assign the variables X and Y to the desires columns.
## Note: Encoding has to be done on your own.
Check test.py to see the use case.
```python
from fastML import fastML

fastML(X, Y, size)
```
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
