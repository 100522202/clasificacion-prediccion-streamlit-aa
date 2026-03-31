import pandas as pd
import joblib
import numpy as np

modelo = joblib.load("modelo_final.joblib")

# 🔹 Instancia 1
instancia1 = pd.DataFrame([{
    'age': 43,
    'job': 'blue-collar',
    'marital': 'single',
    'education': 'primary',
    'default': 'no',
    'balance': -399,
    'housing': 'no',
    'loan': 'yes',
    'contact': 'cellular',
    'day': 28,
    'month': 'jul',
    'campaign': 3,
    'pdays': np.nan,
    'previous': 0,
    'poutcome': 'unknown',
    'duration': 662,
    'contactado_previamente': 0
}])

# 🔹 Instancia 2
instancia2 = pd.DataFrame([{
    'age': 34,
    'job': 'technician',
    'marital': 'married',
    'education': 'secondary',
    'default': 'no',
    'balance': 500,
    'housing': 'yes',
    'loan': 'no',
    'contact': 'cellular',
    'day': 15,
    'month': 'may',
    'campaign': 1,
    'pdays': np.nan,
    'previous': 1,
    'poutcome': 'success',
    'duration': 200,
    'contactado_previamente': 1
}])

print("Predicción instancia 1:", modelo.predict(instancia1))
print("Predicción instancia 2:", modelo.predict(instancia2))