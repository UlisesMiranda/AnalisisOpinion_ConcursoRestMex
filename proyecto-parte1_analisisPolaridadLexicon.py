import operator
import os
import random
import re
import statistics
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, cross_val_score, train_test_split, KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from itertools import cycle
from spacy.lang.es.stop_words import STOP_WORDS
from nltk.corpus import stopwords
import spacy

class ValSet:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

class DataSet:
    def __init__(self, val_set):
        self.val_set = val_set

def cross_validation(elements, k):  # elements = [x_train, y_train]
    val_sets = []
    kf = KFold(n_splits=k)

    for train_i, test_i in kf.split(elements[0]):
        x_train_v, x_test_v = elements[0][train_i], elements[0][test_i]
        y_train_v, y_test_v = elements[1][train_i], elements[1][test_i]

        val_sets.append(ValSet(x_train_v, y_train_v, x_test_v, y_test_v))
        
    data_set = DataSet(val_sets)
    
    return data_set
    

def obtenerDiccionarioEmociones():
    
    lexiconDic = {}
    file = open('SEL_full.txt', 'r', encoding="utf-8")
    for line in file:
    
        palabras = line.split("\t")
        palabras[6] = re.sub('\n', '', palabras[6])
  
        pair = (palabras[6], palabras[5])
  
        if lexiconDic:
            if palabras[0] not in lexiconDic:
                lista = [pair]
                lexiconDic[palabras[0]] = lista
            else:       
                lexiconDic[palabras[0]].append (pair)
        else:
            lista = [pair]
            lexiconDic[palabras[0]] = lista
    file.close()
 
    del lexiconDic['Palabra']; 
    return lexiconDic

def obtenerPolaridades(x_train, y_train, lexiconDic, boundStart, salto):

    features = []
    for opinion in x_train:
    
        valor_alegria = 0.0000
        valor_enojo = 0.0000
        valor_miedo = 0.0000
        valor_repulsion = 0.0
        valor_sorpresa = 0.0
        valor_tristeza = 0.0
        
        op_separada = re.split('\s+', opinion)
  
        dicEmociones = {}
        for palabra in op_separada:
        
            if palabra in lexiconDic:
                puntuaciones = lexiconDic[palabra]
    
                for emocion, valor in puntuaciones:
                    if emocion == 'Alegría':
                        valor_alegria += round(float(valor),4)
                    elif emocion == 'Tristeza':
                        valor_tristeza += round(float(valor),4)
                    elif emocion == 'Enojo':
                        valor_enojo += round(float(valor),4)
                    elif emocion == 'Repulsión':
                        valor_repulsion += round(float(valor),4)
                    elif emocion == 'Miedo':
                        valor_miedo += round(float(valor),4)
                    elif emocion == 'Sorpresa':
                        valor_sorpresa += round(float(valor),4)
      
        dicEmociones['Alegria'] = valor_alegria
        dicEmociones['Tristeza'] = valor_tristeza
        dicEmociones['Enojo'] = valor_enojo
        dicEmociones['Repulsion'] = valor_repulsion
        dicEmociones['Miedo'] = valor_miedo
        dicEmociones['Sorpresa'] = valor_sorpresa
        
        emocionPositivaAcum = dicEmociones['Alegria'] + dicEmociones['Sorpresa']
        emocionNegativaAcum = dicEmociones['Enojo'] + dicEmociones['Miedo'] + dicEmociones['Repulsion'] + dicEmociones['Tristeza']
        
        difPosNeg = emocionPositivaAcum - emocionNegativaAcum
        
        if difPosNeg > boundStart + (salto*3):
            features.append(5)
        elif boundStart + (salto*2) <= difPosNeg and difPosNeg <= boundStart + (salto*3):
            features.append(4)
        elif boundStart + salto <= difPosNeg and difPosNeg < boundStart + (salto*2):
            features.append(3)
        elif boundStart <= difPosNeg and difPosNeg < boundStart + salto:
            features.append(2)
        elif difPosNeg < boundStart:
            features.append(1)
        
            
        # if difPosNeg > -0.0003:
        #     features.append(5)
        # elif -0.3 <= difPosNeg and difPosNeg <= -0.0003:
        #     features.append(4)
        # elif -0.6 <= difPosNeg and difPosNeg < -0.3:
        #     features.append(3)
        # elif -1 <= difPosNeg and difPosNeg < -0.6:
        #     features.append(2)
        # elif difPosNeg <-1:
        #     features.append(1)

    
    y_true = y_train
    y_pred = features   

    accuracy = accuracy_score(y_true, y_pred)
    
    target_names = ['Muy Negativo','Negativo', 'Neutro','Positivo', 'MuyPositivo']
    reporte = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    
    file.write(f"--------UMBRAL: {boundStart} hasta {boundStart + salto*3} con saltos de {salto}\n")
    file.write("Reporte de clasificación:\n")
    file.write(reporte)
    file.write("\n")
    
    return np.array([accuracy, reporte, y_true, y_pred], dtype=object)
             
if 'ReportesClasificacion.txt' in os.listdir('.'):  
    os.remove('./ReportesClasificacion.txt')

nlp = spacy.load("es_core_news_sm")

dfRestMex = pd.read_csv('restMexLematizado.csv')

X = dfRestMex.iloc[:, 1].values
y = dfRestMex.iloc[:, -2].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0, shuffle=True)

lexiconDic = obtenerDiccionarioEmociones()

num_folders = 5
validation_set = cross_validation([X_train, y_train], num_folders)

accuracyList_1 = []
accuracyList_2 = []
accuracyList_3 = []
accuracyList_4 = []

umbralesInicio = [-4, -3, -3.5, -2.5]
salto = 0.35
# umbralesInicio = [-1, -1, -1, -1] 
# salto = 0.5
# umbralesInicio = [-2.5, -2, -1.5, -1] 
# salto = 0.6

print("salto", salto)
    
file = open("ReportesClasificacion.txt", 'a')

for i in range(num_folders):
    file.write(f"---------------Folder {i}:------------\n")
    print("Folder ", i)
    accuracyList_1.append(obtenerPolaridades(validation_set.val_set[i].x_train, validation_set.val_set[i].y_train, lexiconDic, umbralesInicio[0], salto))
    accuracyList_2.append(obtenerPolaridades(validation_set.val_set[i].x_train, validation_set.val_set[i].y_train, lexiconDic, umbralesInicio[1], salto))
    accuracyList_3.append(obtenerPolaridades(validation_set.val_set[i].x_train, validation_set.val_set[i].y_train, lexiconDic, umbralesInicio[2], salto))
    accuracyList_4.append(obtenerPolaridades(validation_set.val_set[i].x_train, validation_set.val_set[i].y_train, lexiconDic, umbralesInicio[3], salto))

accuracyList_1 = np.array(accuracyList_1, dtype=object)
accuracyList_2 = np.array(accuracyList_2, dtype=object)
accuracyList_3 = np.array(accuracyList_3, dtype=object)
accuracyList_4 = np.array(accuracyList_4, dtype=object)
    
accuracy_1_promedio = statistics.mean(accuracyList_1[:, 0])
accuracy_2_promedio = statistics.mean(accuracyList_2[:, 0])
accuracy_3_promedio = statistics.mean(accuracyList_3[:, 0])
accuracy_4_promedio = statistics.mean(accuracyList_4[:, 0])

accuracyPromedios = {
    umbralesInicio[0]: accuracy_1_promedio,
    umbralesInicio[1]: accuracy_2_promedio,
    umbralesInicio[2]: accuracy_3_promedio,
    umbralesInicio[3]: accuracy_4_promedio
}

print("Promedios accuracy: ", accuracyPromedios)

mejorPromedio = max(accuracyPromedios.items(), key=lambda x: x[1])
print("Mejor promedio: ", mejorPromedio)

accuracyFinal = 0

if(mejorPromedio[0] == umbralesInicio[0]):
    accuracyFinal = obtenerPolaridades(X_test, y_test, lexiconDic, umbralesInicio[0], salto)
elif(mejorPromedio[0] == umbralesInicio[1]):
    accuracyFinal = obtenerPolaridades(X_test, y_test, lexiconDic, umbralesInicio[1], salto)
elif(mejorPromedio[0] == umbralesInicio[2]):
    accuracyFinal = obtenerPolaridades(X_test, y_test, lexiconDic, umbralesInicio[2], salto)
elif(mejorPromedio[0] == umbralesInicio[3]):
    accuracyFinal = obtenerPolaridades(X_test, y_test, lexiconDic, umbralesInicio[3], salto)
file.close()
    
print("Accuracy final: ", accuracyFinal[0])
y_true = accuracyFinal[2]
y_pred = accuracyFinal[3]

print (confusion_matrix(y_true, y_pred))
ConfusionMatrixDisplay.from_predictions(y_true, y_pred)

plt.show()

# Define el procesador de lenguaje natural
# vectorizer = CountVectorizer(stop_words=stopwords.words("spanish"))

# # Transforma el texto en una matriz de términos
# X_train = vectorizer.fit_transform(X_train)
# X_test = vectorizer.transform(X_test)

# Calcula la precisión del modelo utilizando validación cruzada

# Regresion logistica 0.71
# clf = LogisticRegression(C=1.0, max_iter=1000)
# accuracy = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
# print("Accuracy regresion logistica: %.2f" % np.mean(accuracy))
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

# Modelo Naive Bayes 0.69
# nb = MultinomialNB()
# accuracy = cross_val_score(nb, X_train, y_train, cv=5, scoring="accuracy")
# print("Accuracy naive bayes: %.2f" % np.mean(accuracy)) 
# nb.fit(X_train, y_train)
# y_pred = nb.predict(X_test)

# Modelo de arbol de decision 0.65
# ad = DecisionTreeClassifier()
# accuracy =  cross_val_score(ad, X_train, y_train, cv=5, scoring="accuracy")
# print("Accuracy arbol de decision: %.2f" % np.mean(accuracy))
# ad.fit(X_train, y_train)
# y_pred = ad.predict(X_test)

# Modelo de red neuronal
# rn = MLPClassifier()
# accuracy =  cross_val_score(rn, X_train, y_train, cv=5, scoring="accuracy")
# print("Accuracy red neuronal: %.2f" % np.mean(accuracy))
# rn.fit(X_train, y_train)
# y_pred = rn.predict(X_test)

# da = {
#     "y_test" : y_test,
#     "y_pred" : y_pred
# }

# prediccion=pd.DataFrame(da)
# prediccion.to_csv("Prediccion.csv")









