# Disaster-Response-Pipeline

## Table of Contents

1. [Overview](#overview)
2. [File Description](description)
3. [Getting Started](#getting-started)
    1. [Dependencies](#dependencies)
    2. [Installation](#installation)
4. [Project Motivation](#project-motivation)
5. [Web Application Demo](#demo)
6. [Author](#author)
7. [Acknowledgments](#ack)

## Overview <a name="overview"></a>
The omnipresence of smartphones has enabled people to call for help in the event of a disaster/emergency in real-time. This project monitors such SOS messages, which can then be forwarded to respective relief organizations for a targeted disaster response.

## Getting Started <a name="getting-started"></a>

### File Description <a name="description"></a>
    Disaster-Response-Pipeline
        ├── data                   
        │   ├── disaster_categories.csv          # Disaster categories dataset for processing
        │   ├── disaster_messages.csv            # Sample disaster messages dataset for processing
        │   └── process_data.py                  # Data cleaning processing script
        |   └── DisasterResponse.db              # Final database, generated as a result of process_data.py
        ├── app     
        │   ├── run.py                           # Flask file that runs the web app
        │   └── templates   
        │       ├── go.html                      # Classification result page of web app
        │       └── master.html                  # Main page of web app
        ├── models
        │   ├── train_classifier.py              # ML model processing script
        |   └── classifier.pkl                   # Trained ML model, generated as a result of train_classifier.py
        ├── webapp-demo.PNG
        └── README.md
    

### Dependencies <a name="dependencies"></a>
*    Python 3.5+
*    Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
*    Natural Language Process Libraries: NLTK
*    SQLlite Database Libraqries: SQLalchemy
*    Model Loading and Saving Library: Pickle, joblib
*    Web App and Data Visualization: Flask, Plotly


### Installation <a name="installation"></a>

Datasets: The set of sample disaster SOS messages can be downloaded from [here](https://github.com/nazianafis/Disaster-Response-Pipeline/blob/main/data/disaster_messages.csv). Disaster categories are available [here](https://github.com/nazianafis/Disaster-Response-Pipeline/blob/main/data/disaster_categories.csv).

#### To run the project:

Run the following commands in the project's root directory.

To run the ETL pipeline that cleans data and stores it in a database:
```
  $ python3 data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```
To run the ML pipeline that trains classifier and saves the model in a pickle file:
```
  $ python3 models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```
To run the web app:
```
  $ python3 app/run.py
```
Go to http://0.0.0.0:3001/ or localhost:3001

### Project Motivation <a name="project-motivation"></a>

For this project, I have tried to build a disaster response web application that can classify SOS messages into different categories like medical supplies, food, or block road. This categorised data can then be used by organizations for providing targeted disaster relief.

### Web Application Demo <a name="demo"></a>

The web application successfully classifies the message "Due to heavy rains, there's flood in Mumbai. Help neeeded" into categories such as aid related, and search and rescue.
 
![demo-image](https://github.com/nazianafis/Disaster-Response-Pipeline/blob/main/webapp-demo.PNG)

## Author <a name="author"></a>
* [Nazia N.](https://github.com/nazianafis)

## Acknowledgements <a name="ack"></a>

* Udacity and Figure Eight for providing the relevant dataset to train the model.
