import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets as dataframes; merge the messages and categories datasets using the common id.
    :param messages_filepath: file path of message dataset.
    :param categories_filepath: file path of categories dataset.
    :return: a dataframe that merge the two datasets.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on=('id'))
    print('Number of rows and columns in the merged file are: {} and {}'.format(df.shape[0], df.shape[1]))
    return df

def clean_data(df):
    """
    Convert category values to just numbers 0 or 1; drop the original categories column from `df`.
    Remove duplicates.
    :param df: The merged dataframe.
    :return: A cleaned dataframe.
    """
    categories = df["categories"].str.split(";", expand=True)
    row = categories.iloc[0]
    category_colnames = [category_name.split('-')[0] for category_name in row.values]
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    df = df.drop(["categories"], axis=1)
    df = pd.concat([df, categories], join='inner', axis=1)
    df = df.drop_duplicates()
    # Remove the column of "child alone" since it has all zeros only
    df = df.drop(['child_alone'], axis=1)
    # The column of "related" has 0, 1, and 2 values. It could be error.
    # Given value 2 in the related field are negligible. Replacing 2 with 1 to consider it a valid response.
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)

    return df

def save_data(df, database_filename):
    """
    Save the clean dataset into an sqlite database.
    :param df: The clean dataset.
    :param database_filename: The file name of the database to be saved.
    """

    engine = create_engine('sqlite:///'+ database_filename)
    table_name = database_filename.replace(".db","") + "_table"
    df.to_sql(table_name, engine, index=False, if_exists='replace')

def main():
    """
    Main function which will conduct the data processing functions.
    First, load datasets of messages and categories.
    Second, clean the dataset.
    Third, save the dataframe into SQL database.
    """
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