from matplotlib import pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

def balancearCorpus(X, y, columnaY):
    # Aplica RandomOverSampler
    ros = RandomOverSampler(random_state=0)
    X, y = ros.fit_resample(X, y[columnaY])
    
    return X, y

print("REGRESION LOGISTICA FRECUENCIAL PROBANDO EL 20% DE LOS DATOS")

# Carga el archivo xlsx en un DataFrame
df_processed = pd.read_csv('RestMex_proyectoParte2.csv')

# Separa el dataset en conjunto de entrenamiento y prueba, utilizando el 80% y el 20% de los datos respectivamente
X = df_processed[['Titulo_lemmatized', 'Opinion_lemmatized']]  # Datos
X = X['Titulo_lemmatized'].values + X['Opinion_lemmatized'].values
y = df_processed[['Polarity', 'Attraction']]  # Clases

#Dividir el conjunto
X_train_org, X_test_org, y_train_org, y_test_org = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

# Inicializa el vectorizador DE FRECUENCIAS
vectorizer = CountVectorizer(binary=False)

# Vectorizamos
X_train_org = vectorizer.fit_transform(X_train_org)
X_test_org = vectorizer.transform(X_test_org)

# ------------------PARA ATRACCION
# Balancear
# x_train_atrr_bal, y_train_atrr_bal = balancearCorpus(X_train_org, y_train_org, "Attraction")

# Inicializa el modelo
model = LogisticRegression(max_iter = 5000, random_state=0, C=0.5)

# Entrena el modelo con el conjunto de entrenamiento
model.fit(X_train_org, y_train_org['Attraction'])

# Realiza la predicción con el conjunto de prueba
y_pred = model.predict(X_test_org)

# Calcula las métricas
acc = accuracy_score(y_test_org['Attraction'], y_pred)
prec = precision_score(y_test_org['Attraction'], y_pred, average='micro')
rec = recall_score(y_test_org['Attraction'], y_pred, average='micro')
f1 = f1_score(y_test_org['Attraction'], y_pred, average='micro')

print("ATTRACTION resultados:")

print(f'Accuracy: {acc:.4f}')
print(f'Precision: {prec:.4f}')
print(f'Recall: {rec:.4f}')
print(f'F-measure: {f1:.4f}')

confusion = ConfusionMatrixDisplay.from_predictions(y_test_org['Attraction'], y_pred)
reporte = classification_report(y_test_org['Attraction'], y_pred, zero_division=0)
print(reporte)
plt.show()
print("")


# --------- PARA POLARIDAD

# Balancear
# x_train_polar_bal, y_train_polar_bal = balancearCorpus(X_train_org, y_train_org, "Polarity")

#Inicializamos el modelo
# FMEASURE 0.49
# model = LogisticRegression(max_iter = 5000, random_state=1, C=1)
# model = LogisticRegression(max_iter = 5000, random_state=1, C=0.5)

# Mas ACCURACY
model = LogisticRegression(max_iter = 5000, random_state=1, C=0.05)

# Entrena el modelo con el conjunto de entrenamiento
model.fit(X_train_org, y_train_org['Polarity'])

# Realiza la predicción con el conjunto de prueba
y_pred = model.predict(X_test_org)

# Calcula las métricas
acc = accuracy_score(y_test_org['Polarity'], y_pred)
prec = precision_score(y_test_org['Polarity'], y_pred, average='micro')
rec = recall_score(y_test_org['Polarity'], y_pred, average='micro')
f1 = f1_score(y_test_org['Polarity'], y_pred, average='micro')

# prec = precision_score(y_test_org['Polarity'], y_pred, average='micro')
# rec = recall_score(y_test_org['Polarity'], y_pred, average='micro')
# f1 = f1_score(y_test_org['Polarity'], y_pred, average='micro')
print("\nPOLARIDAD resultados")

print(f'Accuracy: {acc:.4f}')
print(f'Precision: {prec:.4f}')  
print(f'Recall: {rec:.4f}')
print(f'F-measure: {f1:.4f}')


confusion = ConfusionMatrixDisplay.from_predictions(y_test_org['Polarity'], y_pred)
reporte = classification_report(y_test_org['Polarity'], y_pred, zero_division=0)
print(reporte)
plt.show()
print("")