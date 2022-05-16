# import libraries
import sys
import os
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    INPUT
    messages_filepath - file path of messages data file
    categories_filepath - file path of categories data file
    
    OUTPUT
    the function return a dataframe that merged the messages data and category data after clean-up
    '''
    # load messages / categories dataset
    messages = pd.read_csv(messages_filepath, dtype=str)
    categories = pd.read_csv(categories_filepath, dtype=str)
    
    # drop duplicate rows
    messages.drop_duplicates(inplace=True)
    categories.drop_duplicates(inplace=True)
    
    # merge the messages and categories datasets using the common id
    df = messages.merge(categories, how='outer', on=['id'])
    
    # Split the values in the categories column on the ; character so that each value       # becomes a separate column. The first row of categories dataframe to create column     
    # names for the categories data
    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # extract a list of new column names for categories.
    category_colnames = row.str.slice(start=0, stop=-2)
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(start=-1)

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
        # convert 2 in value to 1 to make the categories binary
        categories[column] = categories[column].replace(2,1)

    # Drop the categories column from the df dataframe since it is no longer needed.
    # Concatenate df and categories data frames.
    df.drop(['categories'], axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join='inner', sort=False)

    return df

def clean_data(df):
    '''
    INPUT
    df - dataframe returned from load_data function
    
    OUTPUT
    the function return a dataframe that removed duplicate rows
    '''    
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filepath):
    '''
    INPUT
    df - dataframe returned from clean_data function
    database_filepath - the name and filepath of a database where the clean dataset to be stored
    
    OUTPUT
    None returned.
    '''
    '''Save the clean dataset into an sqlite database'''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    table_name = os.path.basename(database_filepath).replace('.db','')
    #table_name = database_filepath.replace(".db","")
    df.to_sql(table_name, engine, index=False, if_exists='replace')
 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        #sys.exit()
        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()