# Disaster Response Pipeline Project

### Summary

This project does 3 things.

1. Process messages and categories into a SQLLite database.
2. Use ML to train a classifier for the messages.  This will attempt to classify a message into up to 36 different categories.
3. Host a webapp that gives summary stats about the training data and allows interactive message classification.

### Instructions

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python ./data/process_data.py ./data/disaster_messages.csv ./data/disaster_categories.csv ./data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python ./models/train_classifier.py ./data/DisasterResponse.db ./models/classifier.pkl`

2. Run the following command in the main directory to run your web app.
    `python ./app/run.py`

3. Go to http://localhost:3001/

### Libaries

- python 3.6.6
- plotly 3.3
- pandas 0.23.4
- sqlalchemy 1.2.12
- nltk 3.3.0
- py-xgboost 0.72
- scikit-learn 0.20.0
- jsonschema 2.6.0
- flask 1.0.2

### Files

- app
    - /templates                HTML templates for flask
    - run.py                    Starts the flask webpage
- data
    - disaster_categories.csv   raw category data
    - disaster_messages.csv     raw message data
    - DisasterResponse.db       sqllite: cleaned message/category data stored in messages table
    - process_data.py           Run with two raw files to generate the database (cleaned data)
- models
    - classifier.pkl            Pre-trained classifier, stored as a gzip pickle file
    - train_classifier.py       Run with database file to generate a trained model for use in the webpage