# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT
    file paths of the message and categories files in csv format
    OUTPUT
    a dataframe that contains both datasets
    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')

    return df 

def clean_data(df): 
    '''
    INPUT
    a dataframe with both messages and categories for data cleaning
    OUTPUT
    cleaned dataframe, with new expanding columns for each message category
    '''
    
    # split categories into separate category column
    categories = df['categories'].str.split(';',expand=True)
    new_names = pd.Series(categories.loc[0].values).str.split('-', expand = True)[0].values
    new_names = dict(zip(np.arange(categories.shape[0]), new_names))

    # rename the new splitted columns
    categories = categories.rename(columns = new_names)

    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(np.int)

    # Replace 'categories' column in df with new category columns and drop duplicates    
    df.drop(['categories'], axis = 1, inplace = True)
    df = pd.concat([df, categories], axis=1, join="inner").drop_duplicates()

    # correcing the mislabels to binary
    df.loc[df['related'] > 1,'related'] = 0

    return df


def save_data(df, database_filename):
    '''
    INPUT
    cleaned dataframe and the filepath for the SQL database for saving the dataframe
    OUTPUT
    None
    '''

    engine = create_engine('sqlite:///'+ database_filename)
    table_name = database_filename.replace(".db","") + "_table"
    df.to_sql("_table", engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

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
