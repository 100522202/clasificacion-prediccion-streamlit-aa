# Pipeline de Clasificación y Despliegue en Streamlit - Aprendizaje Automático (AA)

Práctica 1 de la asignatura **Aprendizaje Automático (UC3M)**.

---

## Descripción del Proyecto

Pipeline end-to-end de Machine Learning aplicado a la predicción de suscripciones en campañas bancarias (dataset Bank Marketing).

### Metodología:
* **EDA (Exploratory Data Analysis):** Análisis univariante y multivariante, tratamiento de valores faltantes y transformaciones de variables.
* **Preprocesamiento con Pipelines:** Transformadores personalizados para variables numéricas (`StandardScaler`, `MinMaxScaler`) y categóricas.
* **Modelado:** Comparación de K-Nearest Neighbors (KNN) y Árboles de Decisión (Decision Trees).
* **Validación Robusta:** Evaluación con Holdout y Nested Cross-Validation (validación cruzada anidada) para evitar el data leakage.
* **HPO (Hyperparameter Optimization):** Búsqueda de hiperparámetros óptimos.
* **Despliegue Web:** Aplicación interactiva construida con **Streamlit** para inferencia en tiempo real.

---

## Ejecución de la App Web

```bash
pip install -r requirements.txt
streamlit run mystreamlit.py
```
