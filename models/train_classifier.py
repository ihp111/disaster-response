# import packages
import sys
import os
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sqlalchemy import create_engine
from sklearn.metrics import classification_report
import pickle

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data(database_filepath):
    '''
    INPUT
    database_filepath - file path of the database to read in data from
    
    OUTPUT
    the function returns features, label arrays, and category names
    '''    
    # read in file
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    table_name = os.path.basename(database_filepath).replace('.db','')
    df = pd.read_sql_table(table_name,engine)
    
    # define features and label arrays
    X = df.message.values
    y = df.iloc[:, 4:].values
    category_names = df.iloc[:,4:].columns.values
    return X, y, category_names


def tokenize(text):
    '''
    INPUT
    text - individual messages to tokenize and clean
    
    OUTPUT
    the function returns cleaned tokens
    ''' 
    # text processing
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    '''
    INPUT
    none
    
    OUTPUT
    the function returns a model pipeline including CountVectorizer, transformer, estimators with select parameters and gridsearch object
    ''' 
    # model pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))    
    ])
    
    # define parameters for GridSearchCV
    parameters = {
                'vect__ngram_range': ((1, 1), (1, 2))
                 }
    
    # create gridsearch object and return as final model pipeline
    model = GridSearchCV(pipeline, parameters)
    
    # return model pipeline
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT
    model - a model pipeline returned from the build_model function
    X_test - feature dataset to be used in evaluating a model
    Y_test - label dataset to be used in evaluating a model
    category_names - category_name of the label
    
    OUTPUT
    Label, confusion matrix, accurary, best parameters, classification (precision, recall, f1-score)
    ''' 
    y_pred = model.predict(X_test)
    for i, c in enumerate(category_names):

        labels = np.unique(y_pred[:,i])
        confusion_mat = confusion_matrix(Y_test[:,i], y_pred[:,i])
        accuracy = (y_pred[:,i] == Y_test[:,i]).mean()
        classification = classification_report(Y_test[:,i], y_pred[:,i])

        print(c)
        print("Labels:", labels)
        print("Confusion Matrix:\n", confusion_mat)
        print("Accuracy:", accuracy)
        print("Best Parameters:", model.best_params_)
        print("\nClassification:\n", classification)
        print("\n")


def save_model(model, model_filepath):
    '''
    INPUT
    model - model returned from build_model function
    model_filepath - location of the model
    
    OUTPUT
    pickle file version of the model is created
    ''' 
    # save
    with open(model_filepath,'wb') as f:
        pickle.dump(model,f)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()