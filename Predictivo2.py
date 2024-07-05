import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Cargar los datos desde el archivo parquet
data = pd.read_parquet('modelo_regresion.parquet')

# Reemplazar los valores None o NaN con una cadena vacía
data['review_text'] = data['review_text'].fillna('')

# Convertir el texto a minúsculas
data['review_text'] = data['review_text'].str.lower()

# Realizar análisis de sentimiento utilizando TextBlob
from textblob import TextBlob
data['sentiment'] = data['review_text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Asignar etiquetas de sentimiento
data['sentiment'] = data['sentiment'].apply(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral')

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(data['review_text'], data['sentiment'], test_size=0.2, random_state=42)

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


# Use the trained model to make predictions on new data
new_data = pd.DataFrame({'review_text': ['This restaurant is amazing!','terrible' ]})

# Preprocess the new data
new_data['review_text'] = new_data['review_text'].fillna('')
new_data['review_text'] = new_data['review_text'].str.lower()

# Vectorize the new data
new_data_vectorized = vectorizer.transform(new_data['review_text'])

# Make predictions using the trained model
new_predictions = model.predict(new_data_vectorized)

# Print the predictions
print(new_predictions)