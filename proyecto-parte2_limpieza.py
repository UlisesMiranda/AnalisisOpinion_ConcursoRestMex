import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Inicializa el lematizador
lemmatizer = WordNetLemmatizer()

# Define una función para aplicar el proceso de tokenización y lematización a una cadena de texto
def tokenize_lemmatize(text):
  # Tokeniza el texto
  tokens = word_tokenize(str(text))

  # Lematiza cada token y devuelve la lista de lemas
  lemmas = [lemmatizer.lemmatize(token) for token in tokens]
  return lemmas

# Carga el archivo xlsx en un DataFrame
df = pd.read_excel('Rest_Mex_2022_Sentiment_Analysis_Track_Train.xlsx')

# Aplica el proceso de tokenización y lematización a las columnas 'Titulo' y 'Opinion'
df['Titulo_lemmatized'] = df['Title'].apply(tokenize_lemmatize)
df['Opinion_lemmatized'] = df['Opinion'].apply(tokenize_lemmatize)

# Crea una copia del DataFrame original con las columnas 'Titulo_lemmatized', 'Opinion_lemmatized', 'Polarity' y 'Attraction'
df_processed = df[['Titulo_lemmatized', 'Opinion_lemmatized', 'Polarity', 'Attraction']].copy()

df_processed.to_csv("RestMex_proyectoParte2.csv")



