import sys
import pandas as pd
import re
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    INPUT
    messages_filepath - file path for the message data, in csv format
    categories_filepath - file path for the category data, in csv format

    OUTPUT
    df - A dataframe of combined file data
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = messages.join(categories.set_index('id'), on='id', how='left')

    return df


def clean_data(df):
    """
    INPUT
    df - A dataframe of data to clean

    OUTPUT
    df - A cleaned dataframe
    """

    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';', expand=True)

    # select the first row of the categories dataframe
    row = categories[:1]

    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x.str[0:-2]).values.tolist()

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
        
        # convert column from string to numeric
        categories[column] = categories[column].astype('int32')

    df.drop(['categories'], inplace=True, axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # There is a bug in newer versions of pandas that will add parenthesis, commas, and tick-marks 
    #   to the column name of the 2nd DF; this will fix that if it happens.
    df.rename(columns = lambda x : re.sub(r"['(),]", "", str(x)), inplace=True)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # clean up bad data in the related field
    #   I chose to set them to 0 instead of dropping them.
    df.related[df.related == 2] = 0

    return df

def save_data(df, database_filename):
    """
    Store the dataframe into a sqllite database file for future reference

    INPUT
    df - the dataframe to save
    database_filename - the filename/path to save the dataframe into
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')  


def main():
    """
    The main event.  Uses the command line args to clean data and store into a sqllite database file.
    """
    if len(sys.argv) == 4:

        # Read the command line args
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
              'to as the third argument. \n\nExample: python ./data/process_data.py '\
              './data/disaster_messages.csv ./data/disaster_categories.csv '\
              './data/DisasterResponse.db')


if __name__ == '__main__':
    main()