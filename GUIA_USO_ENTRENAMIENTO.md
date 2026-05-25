# Guía de Uso y Entrenamiento

Esta guía resume cómo preparar el entorno, entrenar los modelos del Proyecto Phoenix y ejecutar inferencia con los artefactos exportados.

## Requisitos

Se recomienda usar **Python 3.9 o superior**.

Para instalar las dependencias principales del proyecto:

```bash
pip install -r requirements.txt
```

Para ejecutar el notebook completo, también pueden ser necesarias dependencias de análisis, explicabilidad y descarga de datos:

```bash
pip install pandas numpy matplotlib scikit-learn xgboost lightgbm shap joblib kaggle
```

## Descarga de datos

Los datos provienen de la competencia IEEE-CIS Fraud Detection en Kaggle. Es necesario tener cuenta de Kaggle, aceptar los términos de la competencia y configurar el archivo `kaggle.json`.

```bash
pip install kaggle
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip -d ieee-fraud-detection/
```

Al finalizar, la carpeta `ieee-fraud-detection/` debe contener:

```text
ieee-fraud-detection/
├── train_transaction.csv
├── train_identity.csv
├── test_transaction.csv
└── test_identity.csv
```

## Estructura esperada

```text
PhoenixProject/
├── OrderedLMPhoenixNew.ipynb
├── app.py
├── requirements.txt
├── ieee-fraud-detection/
│   ├── train_transaction.csv
│   ├── train_identity.csv
│   ├── test_transaction.csv
│   └── test_identity.csv
└── model_export/
    ├── xgb_model.pkl
    ├── lgb_model.pkl
    ├── rf_model.pkl
    ├── imputer.pkl
    └── metadata.json
```

## Entrenamiento

El entrenamiento se realiza ejecutando el notebook `OrderedLMPhoenixNew.ipynb` de principio a fin. Las secciones son interdependientes, por lo que no se deben omitir celdas.

### 1. Carga de datos

El notebook carga los cuatro archivos CSV y une las tablas de transacciones e identidad mediante `TransactionID`. La unión se realiza con `left join` para conservar todas las transacciones.

Salida esperada:

```text
train_data: (590540, 434)
test_data:  (506691, 433)
```

### 2. Análisis exploratorio

La sección de EDA revisa distribución de clases, valores faltantes y correlaciones. El paso crítico es la eliminación greedy de colinealidad, que genera la lista `to_drop` para remover variables redundantes.

### 3. Ingeniería de características

El pipeline genera variables nuevas sin usar la variable objetivo, evitando `data leakage`.

| Función | Propósito |
| --- | --- |
| `frequency_encode` | Reemplaza categorías por su frecuencia relativa. |
| `combine_columns` | Crea identificadores jerárquicos aproximados de usuario. |
| `group_aggregate` | Calcula medias y desviaciones por grupo. |
| `group_nunique` | Cuenta valores únicos para detectar inconsistencias de identidad. |

También se normalizan variables temporales `D`, convirtiendo valores absolutos en desviaciones respecto al comportamiento diario.

### 4. Limpieza

Antes del modelado se aplican tres filtros:

1. Eliminar columnas con más del 90% de valores faltantes.
2. Eliminar columnas incluidas en `to_drop`.
3. Eliminar variables categóricas originales sustituidas por codificación de frecuencia.

Los valores faltantes restantes se reemplazan por `-999`, permitiendo que los modelos de árboles traten la ausencia de información como una señal separada.

### 5. Modelado

La validación usa split temporal:

- 80% de transacciones más antiguas para entrenamiento.
- 20% de transacciones más recientes para validación.

Modelos entrenados:

| Modelo | Estrategia |
| --- | --- |
| XGBoost | Búsqueda aleatoria, regularización y `early_stopping_rounds=50`. |
| LightGBM | Búsqueda aleatoria y `is_unbalance=True`. |
| Random Forest | `SimpleImputer` previo y `class_weight='balanced'`. |
| Ensamble | Combinación ponderada de probabilidades. |

Pesos del ensamble:

```text
XGBoost:       0.4
LightGBM:      0.3
Random Forest: 0.3
```

Tiempo estimado de entrenamiento en CPU: entre **1.5 y 3 horas**, dependiendo del hardware.

### 6. Exportación

La última sección del notebook serializa los artefactos necesarios en `model_export/`:

- `xgb_model.pkl`
- `lgb_model.pkl`
- `rf_model.pkl`
- `imputer.pkl`
- `metadata.json`

El archivo `metadata.json` contiene los pesos del ensamble, la lista de variables en el orden de entrenamiento y el umbral óptimo.

## Ejecución de la aplicación

Para ejecutar la interfaz local de Streamlit:

```bash
streamlit run app.py
```

La aplicación carga los modelos desde `model_export/` y permite consultar predicciones usando los modelos individuales o el ensamble.

## Inferencia con Python

Los modelos exportados permiten clasificar nuevas transacciones sin reentrenar. El punto crítico es que los datos nuevos deben pasar por el mismo pipeline de ingeniería de características usado durante el entrenamiento.

```python
import json
import joblib
import numpy as np

xgb = joblib.load("model_export/xgb_model.pkl")
lgb = joblib.load("model_export/lgb_model.pkl")
rf = joblib.load("model_export/rf_model.pkl")
imp = joblib.load("model_export/imputer.pkl")

with open("model_export/metadata.json", encoding="utf-8") as f:
    meta = json.load(f)

w = meta["ensemble_weights"]
ft = meta["feature_names"]
th = meta["optimal_threshold"]

X = X_new.reindex(columns=ft, fill_value=-999)
X = X.fillna(-999)

p_xgb = xgb.predict_proba(X)[:, 1]
p_lgb = lgb.predict_proba(X)[:, 1]
p_rf = rf.predict_proba(imp.transform(X))[:, 1]

p = (
    w["xgb"] * p_xgb
    + w["lgb"] * p_lgb
    + w["rf"] * p_rf
)

y_pred = (p >= th).astype(int)
```

La variable `p` contiene la probabilidad de fraude por transacción. El umbral óptimo guardado en los metadatos es cercano a `0.69`, seleccionado sobre validación temporal.

## Explicabilidad con SHAP

Para interpretar una predicción individual se puede usar `shap.TreeExplainer` sobre el modelo XGBoost:

```python
import shap

exp = shap.TreeExplainer(xgb)
sv = exp.shap_values(X.iloc[[i]])

shap.waterfall_plot(
    shap.Explanation(
        values=sv[0],
        base_values=exp.expected_value,
        data=X.iloc[i].values,
        feature_names=ft,
    ),
    max_display=20,
)
```

El gráfico de cascada muestra qué variables empujan la predicción hacia mayor o menor riesgo, facilitando auditoría y explicación ante analistas de riesgo.

## Problemas frecuentes

| Problema | Solución |
| --- | --- |
| `FileNotFoundError` al cargar CSV | Verificar que `ieee-fraud-detection/` exista y contenga los cuatro archivos requeridos. |
| Error de memoria en feature engineering | Reducir temporalmente el dataset con una muestra o ejecutar en una máquina con más RAM. |
| `RandomizedSearchCV` muy lento | Reducir `n_iter` de 25 a 10 y `cv` de 3 a 2. |
| `KeyError` al alinear columnas | Confirmar que los datos nuevos pasaron por el mismo pipeline de feature engineering. |
