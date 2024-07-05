import streamlit as st

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
    .centered-text {
        text-align: center;
        color: white; /* Texto blanco */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="centered-text">Modelos de Prediccion para la empresa Taco Bell</h1>', unsafe_allow_html=True)
