

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 18:23:24 2022

@author: elazhari
"""

#APP FLASK (commande : flask run)
# Partie formulaire non utilisée (uniquement appel à l'API)

from flask import Flask, jsonify
import pandas as pd
import pickle

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def load_data():
    '''loading the train and original data'''
    data = pd.read_csv('data/app_no_enconded_featureengineering.csv',
    index_col='SK_ID_CURR', encoding ='utf-8')
    sample = pd.read_csv('data/app_test.csv', index_col='SK_ID_CURR',
                         encoding ='utf-8')
    description = pd.read_csv("data/HomeCredit_columns_description.csv", 
        usecols=['Row', 'Description'], index_col=0,
            encoding= 'unicode_escape')
    #target = data.iloc[:,:-1]
    target = data['TARGET']
    return data, sample, target, description

def load_model():
    '''loading the trained model'''
    pickle_in = open('model/banking_model.md', 'rb')
    clf_x = pickle.load(pickle_in)
    return clf_x

def load_prediction(sample, id, clf):
    '''Predict the default probability for a client using SK_CURR_ID'''
    X=sample.iloc[:,1:]
    #X=sample.iloc[:,2:]
    score = clf.predict_proba(X[X.index == int(id)])
    return score

# App config.
DEBUG = True
app = Flask(__name__)
#app.config.from_object(__name__)
#app.config['SECRET_KEY'] = 'Thisis_mysecret'

data, sample, target, description = load_data()
clf = load_model()
# clf_str_seq = json.dumps(clf)
ids_clients = sample.index.values

@app.route('/')
def index():
    return 'Hello,Web App !!'

@app.route('/credit/<chk_id>', methods=['GET'])
def credit(chk_id):
    #récupération id client depuis argument url
    #chk_id = request.args.get('chk_id', default=1, type=int)
    #calculer prédiction de la probabilité de défaut
    prediction = load_prediction(sample, chk_id, clf)
    # renvoyer la prediction au demandeur
    dict_final = {
        'proba de solva en %' : round(float(prediction[:,0])*100,1) ,
        'proba de default en %': round(float(prediction[:,1])*100 ,1)
        }
    return jsonify(dict_final)

#lancement de l'application
if __name__ == '__main__':
    app.run(debug=True)