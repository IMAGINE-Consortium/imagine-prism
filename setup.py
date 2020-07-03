# -*- coding: utf-8 -*-

"""
Setup file for the *IMAGINE-PRISM* package.

"""


# %% IMPORTS
# Built-in imports
from codecs import open
import re

# Package imports
from setuptools import find_packages, setup


# %% SETUP DEFINITION
# Get the long description from the README file
with open('README.md', 'r') as f:
    long_description = f.read()

# Get the requirements list
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

# Read the __version__.py file
with open('imagine_prism/__version__.py', 'r') as f:
    vf = f.read()

# Obtain version from read-in __version__.py file
version = re.search(r"^_*version_* = ['\"]([^'\"]*)['\"]", vf, re.M).group(1)

# Setup function declaration
setup(name="imagine-prism",
      version=version,
      author="Ellert van der Velden, IMAGINE Consortium",
      author_email='evandervelden@swin.edu.au',
      description=("Integration between IMAGINE and PRISM"),
      long_description=long_description,
      license='GPLv3',
      platforms=['Mac OS-X', 'Linux', 'Unix'],
      python_requires='>=3.5, <4',
      packages=find_packages(),
      package_dir={'imagine_prism': "imagine_prism"},
      include_package_data=True,
      install_requires=requirements,
      zip_safe=False,
      )
