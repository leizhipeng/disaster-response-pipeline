import sys
import os
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
import pickle
import nltk
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sqlalchemy import create_engine

def load_data(database_filepath):
    """
    Load data from database
    :param database_filepath: File path of the database.
    :return X: a dataframe containing independent variables.
            Y: a dataframe containing dependent variables.
            category_names:  List of categories name.
    """
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).replace(".db","") + "_table"
    df = pd.read_sql_table(table_name,engine)
    df = df.drop(['child_alone'], axis=1)
    X = df["message"]
    Y = df[df.columns[4:]]
    category_names = Y.columns

    return X, Y, category_names

def tokenize(text):
    """
    A tokenization function to process the text data.
    :param text: Raw text messages.
    :return clean_tokens : List of tokens extracted from the text messages.
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # Extract all the urls from the provided text
    detected_urls = re.findall(url_regex, text)
    # Replace url with a url placeholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, "urlplaceholder")
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    """
    Build a machine learning pipeline.
    :return cv: return a grid search cross validation.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=1)))
    ])

    parameters = {
        'clf__estimator__criterion': ['gini', 'entropy'],
        'clf__estimator__n_estimators': [5, 10, 20],
        'clf__estimator__min_samples_split': [2, 4, 6]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3, n_jobs=-2)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Report the f1 score, precision and recall for each output category of the dataset.
    :param model: the model to be evaluated.
    :param X_test: independent variables of test dataset.
    :param Y_test: dependent variables of test dataset.
    :param category_names: Category names.
    """
    Y_prediction_test = model.predict(X_test)
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, Y_prediction_test[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, Y_prediction_test[:,i])))


def save_model(model, model_filepath):
    """
    Export the model as a pickle file
    :param model: the model to be saved.
    :param model_filepath: file path of the saved model.
    """
    pickle.dump(model, open(model_filepath, 'wb'))


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