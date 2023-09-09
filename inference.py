import pandas as pd
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import string

def prediction(data_frame):
    data = data_frame
    
    model1 = CatBoostClassifier()
    model2 = CatBoostClassifier()
    model1.load_model('model1.cbm') 
    model2.load_model('model2.cbm')  
    
    model = Word2Vec.load('model.kvmodel')
    data['pr_txt'] = data['pr_txt'].apply(lambda x: str.lower(x))
    data['pr_txt'] = data['pr_txt'].apply(lambda x: re.sub(r'[0-9]+', '', x))
   

    def get_embedding(sentence):
        tokens = word_tokenize(sentence)
        vector = [model.wv[token] for token in tokens if token in model.wv]
        if vector:
            return sum(vector) / len(vector)
        else:
            return [0] * 300  

    data['embedding'] = data['pr_txt'].apply(get_embedding)
    features_test = pd.DataFrame(data['embedding'].tolist())
    test_data = Pool(features_test)
    predictions1 = model1.predict(test_data)
    predictions2 = model2.predict(test_data)
    
    return predictions1.ravel(), predictions2.ravel()

def text_prediction(text):
    model1 = CatBoostClassifier()
    model2 = CatBoostClassifier()
    model1.load_model('model1.cbm') 
    model2.load_model('model2.cbm')  
    model = Word2Vec.load('model.kvmodel')

    text = str.lower(text)
    text = re.sub(r'[0-9]+', '', text)
   
    def get_embedding(sentence):
        tokens = word_tokenize(sentence)
        vector = [model.wv[token] for token in tokens if token in model.wv]
        if vector:
            return sum(vector) / len(vector)
        else:
            return [0] * 300  

    embeddings  = get_embedding(text)
    features_test = pd.DataFrame(embeddings).T
    test_data = Pool(features_test)
    predictions1 = model1.predict(test_data)
    predictions2 = model2.predict(test_data)
    
    return predictions1.ravel(), predictions2.ravel()

    


