import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Prediccion de Deposito Bancario")

# CARGAR MODELO
modelo = joblib.load('modelo_final.joblib')

st.title("Predicción de Depósito Bancario")
st.write("Introduce los datos del cliente:")

# INPUTS
age = st.number_input("Edad", min_value=18, max_value=100, value=30)
job = st.selectbox(
    "Trabajo",
    [
        'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
        'retired', 'self-employed', 'services', 'student', 'technician',
        'unemployed', 'unknown'
    ]
)
marital = st.selectbox("Estado civil", ['married', 'single', 'divorced'])
education = st.selectbox("Educación", ['primary', 'secondary', 'tertiary', 'unknown'])
default = st.selectbox("Crédito por defecto", ['no', 'yes'])
balance = st.number_input("Balance", value=1000)
housing = st.selectbox("Hipoteca", ['no', 'yes'])
loan = st.selectbox("Préstamo", ['no', 'yes'])
contact = st.selectbox("Tipo de contacto", ['cellular', 'telephone', 'unknown'])
day = st.number_input("Día del mes del último contacto", min_value=1, max_value=31, value=1)
month = st.selectbox("Mes del último contacto", ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
campaign = st.number_input("Número de contactos campaña", value=1)

# Usar NaN como en entrenamiento
pdays_input = st.number_input("Días desde último contacto (-1 si nunca)", value=-1)
pdays = np.nan if pdays_input == -1 else pdays_input

previous = st.number_input("Número de contactos previos", min_value=0, value=0)
poutcome = st.selectbox("Resultado de campaña anterior", ['unknown','failure','success','other'])
duration = st.number_input("Duración del último contacto (segundos)", min_value=0, value=100)

# VARIABLE DERIVADA
contactado_previamente = 0 if pdays_input == -1 else 1

# Misma logica que en entrenamiento: education='unknown' -> faltante
education_model = pd.NA if education == 'unknown' else education

# DATAFRAME
data = pd.DataFrame({
    'age': [age],
    'job': [job],
    'marital': [marital],
    'education': [education_model],
    'default': [default],
    'balance': [balance],
    'housing': [housing],
    'loan': [loan],
    'contact': [contact],
    'day': [day],
    'month': [month],
    'campaign': [campaign],
    'pdays': [pdays], 
    'previous': [previous],
    'poutcome': [poutcome],
    'duration': [duration],
    'contactado_previamente': [contactado_previamente]
})

# PREDICCIÓN
if st.button("Predecir"):
    pred = modelo.predict(data)[0]
    prob_yes = None
    if hasattr(modelo, "predict_proba"):
        prob_yes = float(modelo.predict_proba(data)[0][1])

    if pred == 1:
        st.success("El cliente SÍ contratará el depósito")
    else:
        st.error("El cliente NO contratará el depósito")

    if prob_yes is not None:
        st.write(f"Probabilidad estimada de contratar el deposito: {prob_yes:.2%}")
        st.progress(prob_yes)
        if prob_yes >= 0.8:
            st.info("Confianza alta: cliente prioritario para contacto comercial.")
        elif prob_yes >= 0.5:
            st.info("Confianza media: posible cliente interesado, revisar contexto.")
        else:
            st.info("Confianza baja: menor prioridad comercial.")
