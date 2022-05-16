# Disaster Response Pipeline Project

## Installation

This repository was wrtten in Python 3, html, sqlalchemy, numpy, pandas, sci-kit learn, nltk, pickle, plotly, Flask, sqlite3, sys

## Project Motivation

For this project, I built a machine learning pipeline to process a flood of incoming messages and classify each message to an appropriate cateogry to relieve disaster workers' workload. There are 36 different categories in total. The project demonstrates it in a web app where the user can input a message and see if it falls under one of the categories along with a couple of charts.

## File Descriptions

In the 'app' folder:

go.html - code for the web application. Displays the result of what category(s) the message falls under 

master.html - Code for displaying visualizations on the web app. 

run.py - creates visualizations of the messages data

In the 'data' folder:

disaster_messages.csv - contains data of all the messages to be analyzed by the machine learning model. 

categories_messages.csv - contains data of all the categories of the messages to be analyzed by the machine learning model. 

process_data.py - contains the code for all of the data preprocessing of the data sets before being input in to the machine learning pipeline

In the 'models' folder:

train_classifier.py - takes the processed dataframe and tokenizes the messages then trains a classifier with a pipeline and does statistical analysis on it to test the performance of the NLP classifier

classifier.pkl - the model pipeline converted to a pickle file

## Results

The model classifies incoming messages to a proper bucket with a fairly high performance. The performance measures are printed once you run run.py file.

## Licensing, Authors, Acknowledgements

Thank you to Bryan Chambers. I referenced his repository https://github.com/BryanChambers510/Disaster-Pipeline on the description of files used in his README file.

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/