#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

from distutils.core import setup
setup(
  name = 'fastML',         # How you named your package folder (MyLib)
  packages = ['fastML'],   # Chose the same as "name"
  version = '1.0',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'A Python package built with sklearn for running multiple classification algorithms to observe their behaviour in as little as 4 lines. This package drastically makes the work of Data Scientists, AI and ML engineers very easy and fast by saving them the physical stress of writing close to 300 lines of code as they would if not for this package.',   # Give a short description about your library
  author = 'Jerry Buaba',                   # Type in your name
  author_email = 'buabajerry@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/buabaj',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/Team-fastML/fastML/archive/v0.1.tar.gz',    # I explain this later on
  keywords = ['Machine Learning', 'Algorithms', 'Classification', 'Neural Net'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'scikit-learn',
          'pandas',
          'numpy',
          'tensorflow',
          'keras',
          'scipy',
          'joblib',
          'pytz',
          'threadpoolctl'
      ],
  long_description=long_description,
  long_description_content_type='text/markdown',
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)