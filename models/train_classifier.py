import sys
import nltk
import re
import pickle
import pandas as pd
import xgboost as xgb
import gzip

from sqlalchemy import create_engine

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', engine)
    X = df.message.values
    y = df[df.columns.difference(['message', 'original', 'genre', 'id'])].values
    category_names = df.columns.difference(['message', 'original', 'genre', 'id']).values.tolist()

    return X, y, category_names


def tokenize(text):
    stop_words = stopwords.words("english")
    
    # remove non words
    text = re.sub(r"[^a-zA-Z]", " ", text)
    
    # tokenize text
    tokens = word_tokenize(text)
    
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        if tok not in stop_words:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tok = lemmatizer.lemmatize(clean_tok, pos='v')
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize))
                        , ('tfidf', TfidfTransformer())
                        , ('moc', MultiOutputClassifier(xgb.XGBClassifier(
                                 #learning_rate =0.1,
                                 #n_estimators=1000,
                                 reg_alpha=.01,
                                 reg_lambda=.01,
                                 max_depth=9,
                                 min_child_weight=5,
                                 gamma=0.4,
                                 subsample=0.6,
                                 colsample_bytree=0.8,
                                 #objective= 'binary:logistic',
                                 nthread=-1,
                                 #scale_pos_weight=1,
                                 #seed=27
                                )))
                    ])

    # This is purposely limited to keep processing time minimal just for this example
    parameters = {
        'vect__ngram_range': ((1, 2), (1,1))
        ,'vect__max_df': (0.5, 1.0)
        ,'vect__max_features': (None, 1000)
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)

    print("Best Params:")
    for key,val in model.best_params_.items(): 
        print("\t", key, "=>", val)

    for index in range(Y_test.shape[1]):
        print(category_names[index])
        print(classification_report(Y_test[:,index], y_pred[:,index]))


def save_model(model, model_filepath):
    fileObject = gzip.open(model_filepath,'wb') 
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
              './models/train_classifier.py ./data/DisasterResponse.db ./models/classifier.pkl')


if __name__ == '__main__':
    main()