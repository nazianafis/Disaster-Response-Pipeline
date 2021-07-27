# Disaster-Response-Pipeline

## Table of Contents

1. [Overview](#overview)
2. [File Description](description)
3. [Getting Started](#getting-started)
    1. [Dependencies](#dependencies)
    3. [Installation](#installation)
4. [Project Motivation](#project-motivation)
5. [Web Application Demo](#demo)
6. [Author](#author)

## Overview <a name="overview"></a>
The omnipresence of smartphones has enabled people to call for help in the event of a disaster/emergency in real-time. This project programmatically monitors such messages, which can then be forwarded to respective relief organizations for targeted disaster response.

## Getting Started <a name="getting-started"></a>

### File Description <a name="description"></a>
* data: This folder contains all the .csv files, .db file and .py file
    * disaster_categories.csv, disaster_messages.csv: These files inside the data folder contains messages, their genres and different categories they beong to.
    * process_data.py: This code takes as its input csv files containing message data and message categories, and creates an SQLite database containing a merged and cleaned version of this data.
    * disaster.db: This file is the database which is used to fetch data whenever needed.
* models: This folder contains the ML pipeline and the pickle file.
    * train_classifier.py: This code takes the SQLite database produced by process_data.py as an input and uses the data contained within it to train and tune a ML model for categorizing messages. The output is a pickle file containing the fitted model. Test evaluation metrics are also printed as part of the training process.
    * classifier.pkl: This file contains the fitted model so we do not need to fit the model again
* app: This folder contains run.py and templates which are used to run our main web application.

### Dependencies <a name="dependencies"></a>
* Python 3.*
* Libraries: Pandas, Sqlalchemy, Plotly, Pickle
* Flask

### Installation <a name="installation"></a>

Datasets: The set of disaster messages is available and can be downloaded from [here](https://github.com/nazianafis/Disaster-Response-Pipeline/blob/main/data/disaster_messages.csv). Disaster categories are [here](https://github.com/nazianafis/Disaster-Response-Pipeline/blob/main/data/disaster_categories.csv).

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
  $ python3 run.py
```
Go to http://0.0.0.0:3001/ or localhost:3001

### Project Motivation <a name="project-motivation"></a>

For this project, I have tried to build a disaster response web application that can classify SOS messages into different categories like medical supplies, food, or block road. This categorised data can then be used by organizations for providing targeted disaster relief.

### Web Application Demo <a name="demo"></a>

The web application successfully classifies the message "Due to heavy rains, there's flood in Mumbai. Help neeeded" into categories such as aid related, and search and rescue.
 
![demo-image](https://github.com/nazianafis/Disaster-Response-Pipeline/blob/main/webapp-demo.PNG)

## Author <a name="author"></a>
* [Nazia N.](https://github.com/nazianafis)
