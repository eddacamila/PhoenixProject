# Proyecto Phoenix: Detección de Fraude en E-Commerce
## IEEE-CIS Fraud Detection Challenge
### Grupo 2

**Equipo**: Diego Alejandro Irreño - David Fernando Lopez - Juan Camilo López – Edda
Camila Rodríguez.


**Objetivo**: El propósito es construir un modelo predictivo de detección de fraude en transacciones de comercio electrónico utilizando tcnicas de *Machine Learning*. Se busca identificar transacciones fraudulentas con alta precisión para proteger tanto a comerciantes como a consumidores en el ecosistema de pagos digitales.

**Origen de los Datos**: [Datos Vesta Corporation](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

**Dataset**: 590,540 transacciones con 434 variables  
**Desbalance**: 3.5% fraude, 96.5% legí­timo



**Nota**: Nos apoyamos con herramientas IA: Copilot, GPT, Gemini para generación de código y corrección de errores y mejora de redacción. 
**Estructura:** Es un dataset de escala industrial compuesto por dos tablas principales: **Transacciones** (datos de pago, montos y variables anonimizadas) e **Identidad** (contexto del dispositivo y red), conectadas por el `TransactionID`.

**Volumen:** Contiene un total de **590,540 registros** y **434 variables**.

**Desafí­o técnico:** Presenta un alto desbalance, donde solo el **3.5%** de las transacciones son fraudulentas.

**Métrica principal:** El modelo se optimiza utilizando el **AUC-ROC** (Ãrea bajo la curva de la caracterí­stica operativa del receptor). Esta métrica es fundamental en este contexto porque permite evaluar el rendimiento del modelo en datos asimétricos o desbalanceados, donde la exactitud simple no es confiable.

---

### Datos interesantes según el análisis exploratorio previo (Incorporado en este notebook):

El documento destaca que ya identificaron variables crÃ­ticas que "separan" bien el fraude:

* El **Producto C** tiene la tasa más alta de fraude (~12%).

* Las **tarjetas de crédito** son m-as vulnerables (~7%) que las de dÃ©bito.

* El uso de **dispositivos móviles** duplica el riesgo de fraude en comparaciÃ³n con computadoras de escritorio (*desktop*).
