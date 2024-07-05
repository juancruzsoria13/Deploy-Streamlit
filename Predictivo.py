import streamlit as st
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Función para cargar datos
@st.cache_data
def load_data():
    ruta1 = "df_modelo2.parquet"
    
    dfmodelo = pd.read_parquet(ruta1)
    
    return dfmodelo

# Cargar datos
dfmodelo = load_data()

# Convertir la columna review_date a formato numérico
dfmodelo['review_date'] = pd.to_datetime(dfmodelo['review_date']).dt.strftime('%Y%m%d').astype(int)

# Filtrar los datos para los restaurantes Taco Bell
taco_bell_data = dfmodelo[dfmodelo['business_name'].str.contains('Taco Bell', case=False)]

# Seleccionar las características y la variable objetivo
features = ['review_date', 'city']
target = 'business_rating'

# Codificar la columna 'city' utilizando one-hot encoding
encoder = OneHotEncoder(handle_unknown='ignore')
city_encoded = encoder.fit_transform(taco_bell_data[['city']])

# Convertir el resultado de one-hot encoding a un DataFrame
city_encoded_df = pd.DataFrame(city_encoded.toarray(), columns=encoder.get_feature_names_out(['city']))

# Concatenar las características codificadas con las otras características
taco_bell_data_encoded = pd.concat([taco_bell_data[['review_date']].reset_index(drop=True), city_encoded_df], axis=1)

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(taco_bell_data_encoded, taco_bell_data[target], test_size=0.2, random_state=42)

# Crear el modelo de regresión XGBoost
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)

#####

# Listado de ciudades de Florida
ciudades_florida = taco_bell_data['city'].unique()

st.markdown(
    """
    <style>
    .main {
        background-color: #800080;  /* Fondo morado */
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

st.title("Predicción de Ratings para Taco Bell")
st.header("Predicción de rating para una ciudad y año específicos")

# Interfaz para ingresar ciudad y año
ciudad = st.selectbox("Selecciona una ciudad", ciudades_florida)
anio = st.number_input("Ingresa el año", min_value=2023, max_value=2030, step=1)

if st.button("Predecir"):
    if ciudad and anio:
        new_data = pd.DataFrame({'review_date': [int(f"{anio}0101")], 'city': [ciudad]})
        new_city_encoded = encoder.transform(new_data[['city']])
        new_city_encoded_df = pd.DataFrame(new_city_encoded.toarray(), columns=encoder.get_feature_names_out(['city']))

        new_data_encoded = pd.concat([new_data[['review_date']], new_city_encoded_df], axis=1)

        missing_cols = set(X_train.columns) - set(new_data_encoded.columns)
        for c in missing_cols:
            new_data_encoded[c] = 0
        new_data_encoded = new_data_encoded[X_train.columns]

        new_predictions = xgb_model.predict(new_data_encoded)
        
        prediccion_redondeada = round(new_predictions[0], 2)

        st.write(f"Predicción del rating de Taco Bell en {ciudad} para el año {anio}: {prediccion_redondeada}")

