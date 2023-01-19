import pickle
import re
import pandas as pd
import numpy as np
from sklearn import preprocessing
from spacy.lang.es.stop_words import STOP_WORDS
import spacy

nlp = spacy.load("es_core_news_sm")

def lematizacion(texto):

    doc = nlp(texto)
    palabras = ""
    
    for token in doc:
        if not (token.is_punct | token.is_stop) and token.orth_.isalpha():
            lemma = token.lemma_.lower()
            palabras += lemma + " "
            
    return palabras

df = pd.read_excel("Rest_Mex_2022_Sentiment_Analysis_Track_Train.xlsx")

# Obtenemos los valores columnas de tittle y opinion
X = df.iloc[:, :2].values
# Obtenemos los valores columnas de Polarity y attraction
Y = df.iloc[:, 2:].values

#Categorizamos la columna atracction
# laEncoder = preprocessing.LabelEncoder()
# Y[:,-1] = laEncoder.fit_transform(Y[:,-1])

# Lematizamos las columnas de title y opinion
titulosOpinionesLematizados = []
for i in range(len(X)):
    tituloOpinion_Unida = str(X[i][0]) + "." + str(X[i][1])
    textoLematizado = lematizacion(tituloOpinion_Unida)
    titulosOpinionesLematizados.append(textoLematizado)
    
#Guarda el dataset en csv    
data = {
    "Titulo y opinion": titulosOpinionesLematizados,
    "Polaridad": Y[:, -2],
    "Atraccion": Y[:, -1],
}
dfLemTitulosOpiniones = pd.DataFrame(data)
dfLemTitulosOpiniones.to_csv("restMexLematizado-parte2.csv")

#Guarda el dataset en pickle
# dataset_file = open ('restMexLematizado.pkl','wb')
# pickle.dump(dfLemTitulosOpiniones, dataset_file)
# dataset_file.close()

# dataset_file = open ('dataset.pkl','rb')
# my_data_set_pickle = pickle.load(dataset_file)
# print ("-----------------------------------------------")
# print (*my_data_set_pickle.test_set.X_test)
