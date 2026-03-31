import pandas as pd
import joblib
import numpy as np

modelo = joblib.load("modelo_final.joblib")

# Instancia 1
instancia1 = pd.DataFrame([{
    'age': 43,
    'job': 'management',
    'marital': 'married',
    'education': 'tertiary',
    'default': 'no',
    'balance': 78,
    'housing': 'yes',
    'loan': 'no',
    'contact': 'cellular',
    'day': 21,
    'month': 'nov',
    'campaign': 1,
    'pdays': 109,
    'previous': 1,
    'poutcome': 'other',
    'duration': 36,
    'contactado_previamente': 1
}])

# Instancia 2
instancia2 = pd.DataFrame([{
    'age': 34,
    'job': 'housemaid',
    'marital': 'married',
    'education': 'secondary',
    'default': 'no',
    'balance': 0,
    'housing': 'yes',
    'loan': 'no',
    'contact': 'unknown',
    'day': 30,
    'month': 'oct',
    'campaign': 1,
    'pdays': np.nan,
    'previous': 0,
    'poutcome': 'unknown',
    'duration': 154,
    'contactado_previamente': 0
}])

# Instancia 3
instancia3 = pd.DataFrame([{
    'age': 54,
    'job': 'technician',
    'marital': 'married',
    'education': 'secondary',
    'default': 'no',
    'balance': 3323,
    'housing': 'yes',
    'loan': 'yes',
    'contact': 'cellular',
    'day': 8,
    'month': 'apr',
    'campaign': 3,
    'pdays': np.nan,
    'previous': 0,
    'poutcome': 'unknown',
    'duration': 59,
    'contactado_previamente': 0
}])

# Instancia 4 
instancia4 = pd.DataFrame([{
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

# Instancia 5
instancia5 = pd.DataFrame([{
    'age': 35,
    'job': 'blue-collar',
    'marital': 'married',
    'education': 'secondary',
    'default': 'no',
    'balance': 262,
    'housing': 'no',
    'loan': 'no',
    'contact': 'cellular',
    'day': 15,
    'month': 'mar',
    'campaign': 1,
    'pdays': 181,
    'previous': 3,
    'poutcome': 'success',
    'duration': 427,
    'contactado_previamente': 1
}])

print("Predicción instancia 1:", modelo.predict(instancia1))
print("Predicción instancia 2:", modelo.predict(instancia2))
print("Predicción instancia 3:", modelo.predict(instancia3))
print("Predicción instancia 4:", modelo.predict(instancia4))
print("Predicción instancia 5:", modelo.predict(instancia5))