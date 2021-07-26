# Disaster-Response-Pipeline

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
    1. [Dependencies](#dependencies)
    2. [Installation](#installation)
3. [Project Motivation](#project-motivation)
4. [Web Application Demo](#demo)
3. [Author](#author)

## Overview <a name="overview"></a>
The omnipresence of smartphones has enabled people to call for help in the event of a disaster/emergency in real-time. This project programmatically monitors such messages, which can then be forwarded to respective relief organizations for targeted disaster response.

## Getting Started <a name="getting-started"></a>

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
