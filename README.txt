CREDIT CARD FRAUD DETECTION WEBSITE
===================================

This project turns the supplied ML notebooks into a Flask website.

STEP 1 - Install Python
-----------------------
Install Python 3.11 or newer from python.org.
During installation on Windows, tick:
    "Add Python to PATH"

STEP 2 - Open this project
--------------------------
Open this folder in VS Code.

STEP 3 - Open the VS Code terminal
----------------------------------
Terminal > New Terminal

STEP 4 - Create a virtual environment
-------------------------------------
Windows:
    py -m venv venv
    venv\Scripts\activate

If "py" does not work, try:
    python -m venv venv
    venv\Scripts\activate

STEP 5 - Install libraries
--------------------------
    pip install -r requirements.txt

STEP 6 - Add the dataset
------------------------
Copy your "creditcard.csv.zip" file into:
    data/

So it should look like:
    data/creditcard.csv.zip

Do NOT unzip it.

STEP 7 - Train the models
-------------------------
Run:
    python train_models.py

Wait until it says:
    Training complete!

You should then see:
    models/decision_tree.joblib
    models/knn.joblib
    models/logistic_regression.joblib
    models/svm.joblib
    models/metrics.json

STEP 8 - Start the website
--------------------------
Run:
    python app.py

You should see a message containing:
    http://127.0.0.1:5000

Open that address in Chrome.

IMPORTANT
---------
The website is for your academic ML project/demo.
It is not a real banking fraud-decision system.
