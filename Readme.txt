Pipenv is creates a virtual container and installs all the dependencies in the pipfile that are required for the Shiny app. 

In command prompt:

To install pipenv

$ pip install pipenv

To install dependencies navigate to the folder with app and pipfile and use the command:

$ pipenv install

To run the App
Method-1: Use the following command to run the file directly

$ pipenv run python-file-name.py

Method-2: Initiate a virtual environment using the command:

$ pipenv shell

This opens the virtual environment and then run the python file using the following command:

>> python python-file-name.py 