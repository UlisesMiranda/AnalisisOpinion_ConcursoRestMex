import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE, RandomOverSampler

def balancearCorpus(X, y, columnaY):
    # Aplica RandomOverSampler
    ros = RandomOverSampler(random_state=0)
    X, y = ros.fit_resample(X, y)
    
    return X, y

print("NAIVE BAYES MULTINOMIAL FRECUENCIAS")

# Carga el archivo xlsx en un DataFrame
df_processed = pd.read_csv('RestMex_proyectoParte2.csv')

X = df_processed[['Titulo_lemmatized', 'Opinion_lemmatized']]  # Datos
X = X['Titulo_lemmatized'].values + X['Opinion_lemmatized'].values
y = df_processed[['Polarity', 'Attraction']]  # Clases

# Inicializa el modelo
model = MultinomialNB()

# Inicializa el conjunto de validación
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# ------------------PARA ATRACCION
# Separa el dataset en conjunto de entrenamiento y prueba, utilizando el 80% y el 20% de los datos respectivamente
X_train, X_test, y_train, y_test = train_test_split(X, y['Attraction'].values, test_size=0.2, random_state=0, shuffle=True)

# Inicializa las variables para almacenar las métricas
acc_list_Atttraction = []
prec_list_Atttraction = []
rec_list_Atttraction = []
f1_list_Atttraction = []

# Itera sobre los pliegues con el 80% de los datos
for train_index, val_index in kf.split(X_train):
    print("Pliegue de attraction")
    # Obtiene los conjuntos de entrenamiento y validación para el pliegue actual
    X_train_val = X_train[train_index]
    X_test_val = X_train[val_index]
    
    y_train_val = y_train[train_index]
    y_test_val = y_train[val_index]
    
    # Balancear
    # X_train_val, y_train_val = balancearCorpus(X_train_val, y_train_val)
    
    # Inicializa el vectorizador DE FRECUENCIAS
    vectorizer = CountVectorizer(binary=False)
    
    # Vectorizamos
    X_train_val = vectorizer.fit_transform(X_train_val)
    X_test_val = vectorizer.transform(X_test_val)

    # Entrena el modelo con el conjunto de entrenamiento
    model.fit(X_train_val, y_train_val)

    # Realiza la predicción con el conjunto de validación
    y_pred = model.predict(X_test_val)

    # Calcula las métricas
    acc = accuracy_score(y_test_val, y_pred)
    prec = precision_score(y_test_val, y_pred, average='micro')
    rec = recall_score(y_test_val, y_pred, average='micro')
    f1 = f1_score(y_test_val, y_pred, average='micro')

    # Agrega las métricas a las listas
    acc_list_Atttraction.append(acc)
    prec_list_Atttraction.append(prec)
    rec_list_Atttraction.append(rec)
    f1_list_Atttraction.append(f1)
    
# Calcula el promedio de las métricas
acc_mean_Atttraction = sum(acc_list_Atttraction) / len(acc_list_Atttraction)
prec_mean_Atttraction = sum(prec_list_Atttraction) / len(prec_list_Atttraction)
rec_mean_Atttraction = sum(rec_list_Atttraction) / len(rec_list_Atttraction)
f1_mean_Atttraction = sum(f1_list_Atttraction) / len(f1_list_Atttraction)

print("ATTRACTION resultados:")

print(f'Accuracy promedio: {acc_mean_Atttraction}')
print(f'Precision promedio: {prec_mean_Atttraction}')
print(f'Recall promedio: {rec_mean_Atttraction}')
print(f'F-measure promedio: {f1_mean_Atttraction}')

# --------- PARA POLARIDAD
# Separa el dataset en conjunto de entrenamiento y prueba, utilizando el 80% y el 20% de los datos respectivamente
X_train, X_test, y_train, y_test = train_test_split(X, y['Polarity'].values, test_size=0.2, random_state=0, shuffle=True)

# Inicializa las variables para almacenar las métricas
acc_list_Polarity = []
prec_list_Polarity = []
rec_list_Polarity = []
f1_list_Polarity = []

print("")

# Itera sobre los pliegues
for train_index, val_index in kf.split(X_train):
    print("Pliegue de polaridad")
    # Obtiene los conjuntos de entrenamiento y validación para el pliegue actual
    X_train_val = X_train[train_index]
    X_test_val = X_train[val_index]
    
    y_train_val = y_train[train_index]
    y_test_val = y_train[val_index]
    
    # Balancear
    # X_train_val, y_train_val = balancearCorpus(X_train_val, y_train_val)
    
    # Inicializa el vectorizador DE FRECUENCIAS
    vectorizer = CountVectorizer(binary=False)
    
    # Vectorizamos
    X_train_val = vectorizer.fit_transform(X_train_val)
    X_test_val = vectorizer.transform(X_test_val)
    
    # Entrena el modelo con el conjunto de entrenamiento
    model.fit(X_train_val, y_train_val)

    # Realiza la predicción con el conjunto de validación
    y_pred = model.predict(X_test_val)

    # Calcula las métricas
    acc = accuracy_score(y_test_val, y_pred)
    prec = precision_score(y_test_val, y_pred, average='micro')
    rec = recall_score(y_test_val, y_pred, average='micro')
    f1 = f1_score(y_test_val, y_pred, average='micro')

    # Agrega las métricas a las listas
    acc_list_Polarity.append(acc)
    prec_list_Polarity.append(prec)
    rec_list_Polarity.append(rec)
    f1_list_Polarity.append(f1)
    
# Calcula el promedio de las métricas
acc_mean_Polarity = sum(acc_list_Polarity) / len(acc_list_Polarity)
prec_mean_Polarity = sum(prec_list_Polarity) / len(prec_list_Polarity)
rec_mean_Polarity = sum(rec_list_Polarity) / len(rec_list_Polarity)
f1_mean_Polarity = sum(f1_list_Polarity) / len(f1_list_Polarity)

print("\nPOLARIDAD resultados")

print(f'Accuracy promedio: {acc_mean_Polarity}')
print(f'Precision promedio: {prec_mean_Polarity}')  
print(f'Recall promedio: {rec_mean_Polarity}')
print(f'F-measure promedio: {f1_mean_Polarity}')

print("\nResultados generales:")
print("Accuracy Attraction: ", acc_mean_Atttraction)
print("Accuracy Polarity: ", acc_mean_Polarity)
