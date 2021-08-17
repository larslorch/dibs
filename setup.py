'''
This file allows calling contents of project, i.e. `./dibs/...`
from sibling folders, such as e.g. `./tests/test_script.py`

Simply add this file and execute

    pip install -e .

The -e stands for editable state. All the edits made to .py files will be automatically 
included in the installed package.

Every subdirectory of the package folder `./dibs/` has to have its own (empty) `__init__.py` file.

'''

from setuptools import setup, find_packages
setup(name='dibs', version='1.0', packages=find_packages())
