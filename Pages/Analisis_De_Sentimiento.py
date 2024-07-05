import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from textblob import TextBlob

# Función para cargar datos
@st.cache_data
def load_data():
    ruta1 = "df_modelo2.parquet"
    dfmodelo = pd.read_parquet(ruta1)
    return dfmodelo

# Cargar datos
dfmodelo = load_data()

# Reemplazar los valores None o NaN con una cadena vacía
dfmodelo['review_text'] = dfmodelo['review_text'].fillna('')

# Convertir el texto a minúsculas
dfmodelo['review_text'] = dfmodelo['review_text'].str.lower()

# Realizar análisis de sentimiento utilizando TextBlob
dfmodelo['sentiment'] = dfmodelo['review_text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Asignar etiquetas de sentimiento
dfmodelo['sentiment'] = dfmodelo['sentiment'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(dfmodelo['review_text'], dfmodelo['sentiment'], test_size=0.2, random_state=42)

# Crear el vectorizador TF-IDF para convertir el texto en vectores numéricos para el modelo
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Crear el modelo de clasificación Naive Bayes Multinomial para análisis
model = MultinomialNB()

# Entrenar el modelo de clasificación
model.fit(X_train_vectorized, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test_vectorized)

# Mostrar el reporte de clasificación en la aplicación
st.title("Análisis de Sentimientos de Reseñas")

# Entrada de datos del usuario
st.header("Predicción de Sentimientos para Nuevas Reseñas")
user_input = st.text_area("Introduce una nueva reseña para predecir su sentimiento:")

if st.button("Predecir Sentimiento"):
    if user_input:
        # Preprocesar la nueva entrada
        new_data = pd.DataFrame({'review_text': [user_input]})
        new_data['review_text'] = new_data['review_text'].fillna('')
        new_data['review_text'] = new_data['review_text'].str.lower()

        # Vectorizar la nueva entrada
        new_data_vectorized = vectorizer.transform(new_data['review_text'])

        # Realizar la predicción usando el modelo entrenado
        new_prediction = model.predict(new_data_vectorized)
        
        st.write(new_prediction[0])
    else:
        st.write("Por favor, introduce una reseña para hacer la predicción.")


st.markdown(
    """
    <style>
    .main {
        background-color: #800080;  /* Fondo morado */
        color: white;  /* Texto en blanco */
    }
    .stButton>button {
        color: #800080;  /* Texto morado */
        background-color: white;  /* Botón blanco */
    }
    .stSelectbox, .stNumberInput, .stTextInput {
        background-color: white;
        color: #800080;  /* Morado */
    }
    .stTextInput>div>input, .stNumberInput>div>input {
        background-color: white;
        color: #800080;  /* Morado */
    }
    .stTitle, .stHeader {
        color: white;  /* Títulos en blanco */
    }
    h1, h2 {
        color: white;  /* Títulos y encabezados en blanco */
    }
    </style>
    """,
    unsafe_allow_html=True
)