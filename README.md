# Disaster Response Pipeline Project
* Udacity Data Scientist Nanodegree project: Disaster Response Pipeline

![intro figure](https://github.com/leizhipeng/disaster-response-pipeline/blob/main/figures/intro.png?raw=true)

## Table of Contents
1. [Introduction](#introduction)
2. [File Structure](#filestructure)
3. [Instructions](#instructions)
4. [Dataset](#dataset)
5. [ML Pipeline](#pipeline)
6. [Flask Web App](#flask)

<a name="instroduction"></a>
## Introduction
This Project aims to analyze disaster data, containing real messages that were sent during disaster events. We create a machine learning pipeline to categorize these disaster events so that the messages can be sent to an appropriate disaster relief agency.


<a name="filestructure"></a>
## File Structure:
    app
    | - template
    | |- master.html                # main page of web app
    | |- go.html                    # classification result page of web app
    |- run.py                       # Flask file that runs app
    data
    |- disaster_categories.csv      # data to process
    |- disaster_messages.csv        # data to process
    |- process_data.py              # code for processing data with ETL pipeline
    |- InsertDatabaseName.db        # database to save clean data to
    models
    |- train_classifier.py          # code for obtaining a message classifier with ML pipeline
    |- classifier.pkl               # saved model
    README.md

<a name="instructions"></a>
## Instructions:
* Install dependencies, including NumPy, SciPy, Pandas, Sciki-Learn, NLTK, SQLalchemy, Pickle, Flask, Plotly.
* Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
* Run the following command in the app's directory to run your web app.
    `python run.py`
* Go to http://0.0.0.0:3001/


<a name="dataset"></a>
## Dataset
The messages and categories datasets are provided. We create an **ETL pipeline** to load the datasets, merge the two datasets, clean the data, and store it in a SQLite database.  

![genres figure](https://github.com/leizhipeng/disaster-response-pipeline/blob/main/figures/genres.png?raw=true)
![categories figure](https://github.com/leizhipeng/disaster-response-pipeline/blob/main/figures/categories.png?raw=true)


<a name="pipeline"></a>
## Machine Learning Pipeline
A machine learning pipeline is built to process the message texts and predict the categories. 

![mlpipeline figure](https://github.com/leizhipeng/disaster-response-pipeline/blob/main/figures/mlpipeline.png?raw=true)

<a name="flask"></a>
## Flask Web App
We set up a Flask web app for visualizing data, handling user query, and displaying results.

![query figure](https://github.com/leizhipeng/disaster-response-pipeline/blob/main/figures/inquiry.png?raw=true)
