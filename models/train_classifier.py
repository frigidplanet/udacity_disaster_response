import sys
import nltk
import re
import pickle
import pandas as pd

from sqlalchemy import create_engine

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

def load_data(database_filepath="YourDatabaseName.db"):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', engine)
    X = df.message.values
    y = df[df.columns.difference(['message', 'original', 'genre', 'id'])].values
    category_names = df.columns.difference(['message', 'original', 'genre', 'id']).values.tolist()

    return X, y, category_names


def tokenize(text):
    stop_words = stopwords.words("english")
    
    # remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # tokenize text
    tokens = word_tokenize(text)
    
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        if tok not in stop_words:
            #clean_tok = re.sub(r"[^a-zA-Z0-9]", " ", tok)
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tok = lemmatizer.lemmatize(clean_tok, pos='v')
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize))
                        , ('tfidf', TfidfTransformer())
                        , ('moc', MultiOutputClassifier(OneVsRestClassifier(LinearSVC(random_state=42))))
                    ])

    parameters = {
        'vect__ngram_range': ((1, 2), (1,1))
        ,'vect__max_df': (.25, 0.5)
        ,'vect__max_features': (None, 1000)
        ,'tfidf__use_idf': (True, False)
        ,'tfidf__smooth_idf': (True, False)
        #,'moc__estimator__estimator__dual': (True, False)
        #,'moc__estimator__estimator__max_iter': (1000, 1500)
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)

    for index in range(Y_test.shape[1]):
        print(category_names[index])
        print(classification_report(Y_test[:,index], y_pred[:,index]))


def save_model(model, model_filepath="your_model_name.pkl"):
    
    fileObject = open(model_filepath,'wb') 
    pickle.dump(model, fileObject)  
    fileObject.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        nltk.download(['punkt', 'stopwords', 'wordnet'])

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