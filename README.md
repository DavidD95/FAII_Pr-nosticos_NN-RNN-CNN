# Predicción de Brotes de Dengue en Cali, Colombia
### Un Enfoque Competitivo con Machine Learning para un Problema de Salud Pública

---

## 1. Introducción y Contexto del Problema

El dengue representa un desafío significativo y persistente para la salud pública en la ciudad de Cali, Colombia, que presenta una de las tasas de incidencia más altas del país. La capacidad de anticipar brotes futuros a nivel de barrio permite a las autoridades sanitarias pasar de una gestión reactiva a una **estrategia proactiva**, optimizando la asignación de recursos para campañas de prevención, fumigación y preparación de centros de salud.

## 2. Objetivo del Proyecto

El objetivo principal de este proyecto era desarrollar y evaluar un pipeline de Machine Learning de alta precisión para **predecir el número de casos de dengue semanales por barrio (`id_bar`)**. El proyecto se desarrolló en el marco de una competencia, con el objetivo final de alcanzar la puntuación más alta posible en el leaderboard privado, utilizando el **Error Cuadrático Medio (MSE)** como métrica de evaluación principal.

## 3. Metodología y Flujo de Trabajo

El proyecto siguió un flujo de trabajo iterativo y estructurado, comenzando con modelos base y avanzando hacia técnicas más sofisticadas para maximizar el rendimiento.

### Fase I: Análisis Exploratorio de Datos (EDA)

El análisis inicial de los datos históricos (2015-2021) fue fundamental y reveló varios patrones clave:
* **Ciclos Epidémicos:** Se identificaron claros ciclos multianuales con años de brotes intensos (ej. 2016, 2020) seguidos de períodos de calma.
* **Fuerte Estacionalidad:** Dentro de cada año, los casos de dengue alcanzan su pico máximo consistentemente en la primera mitad del año.
* **El Hallazgo Clave:** Se descubrió un **retardo temporal (lag) de aproximadamente 8 a 16 semanas** entre los picos de precipitación (lluvia) y los picos de casos de dengue. Esta fue la hipótesis central sobre la que se construyeron los modelos posteriores.

### Fase II: Torneo de Redes Neuronales

Se implementó y evaluó sistemáticamente una serie de 5 arquitecturas de redes neuronales para establecer un benchmark y entender la complejidad del problema:
1.  **MLP (Perceptrón Multicapa):** Estableció un sólido baseline inicial (MSE de Validación: 2.93).
2.  **CNN (Red Convolucional):** Sufrió de sobreajuste y tuvo un rendimiento inferior.
3.  **SimpleRNN:** Tuvo un rendimiento estable pero no pudo superar al MLP, probablemente debido al problema del gradiente evanescente.
4.  **LSTM y GRU:** Demostraron ser las arquitecturas de redes neuronales superiores, gracias a su capacidad para manejar dependencias a largo plazo. La **LSTM** se coronó como la mejor de este grupo, con un MSE de Validación de **2.51**.

### Fase III: Estrategia Competitiva y Modelo Ganador

Aunque la LSTM mejoró el baseline, la diferencia con los líderes de la competencia indicaba la necesidad de un enfoque más potente. Se implementó una estrategia avanzada basada en tres pilares:

1.  **Transformación del Objetivo:** Se aplicó una transformación logarítmica (`log1p`) a la variable `dengue` para estabilizar el entrenamiento y reducir el impacto de los valores atípicos de los brotes.
2.  **Ingeniería de Características Masiva:** Se crearon cientos de nuevas características, incluyendo múltiples lags (4, 8, 12, 16, 24 semanas) y una variedad de estadísticas móviles (medias, medianas, std, max) para las variables climáticas y de dengue.
3.  **Cambio a LightGBM:** Se reemplazó la arquitectura de red neuronal por un modelo de **Boosting de Gradiente (LightGBM)**, el cual es estado del arte para datos tabulares estructurados, como el que creamos con la ingeniería de características.

Esta estrategia final resultó en una mejora drástica del rendimiento.

## 4. Resultados

La evolución del rendimiento a lo largo del proyecto demuestra el poder de un enfoque iterativo:

| Modelo / Estrategia | MSE de Validación (OOF*) | Score Público (Test Set) |
| :------------------ | :----------------------: | :----------------------: |
| MLP (Baseline)      |           2.93           |          ~4.10           |
| LSTM (Mejor NN)     |         **2.51** |          ~4.10           |
| **LGBM (Estrategia Final)** |         **~0.62** |       **0.61899** |

*OOF (Out-of-Fold) es el score de nuestra validación cruzada, una estimación robusta del rendimiento.*

El modelo final, basado en LightGBM y una ingeniería de características avanzada, nos posicionó en el **3er lugar del leaderboard**, muy cerca de los dos primeros puestos.

## 5. Estructura del Repositorio

* `/data`: Contiene los archivos de datos originales (`train.parquet`, `test.parquet`, `sample_submission.csv`).
* `dengue_prediction_notebook.ipynb`: El notebook de Jupyter/Colab que contiene todo el código, desde el EDA hasta la generación del submission final.
* `submission.csv`: El archivo de entrega final generado por el notebook.
* `README.md`: Este archivo de resumen.

## 6. Cómo Replicar el Proyecto

1.  Clonar el repositorio:
    ```bash
    git clone [URL-DEL-REPOSITORIO]
    ```
2.  Instalar las dependencias:
    ```bash
    pip install pandas numpy scikit-learn plotly lightgbm
    ```
3.  Ejecutar el notebook `dengue_prediction_notebook.ipynb` en un entorno de Jupyter o Google Colab. Asegúrate de que los archivos de datos estén en la ubicación correcta.

## 7. Conclusión y Futuras Mejoras

Este proyecto demuestra con éxito la creación de un modelo de machine learning de alto rendimiento para un problema real de salud pública. La clave del éxito no residió en un único "modelo mágico", sino en la **sinergia entre un análisis de datos profundo, una ingeniería de características creativa y la selección de la arquitectura de modelo adecuada para la representación de los datos.**

Para buscar el primer lugar, los próximos pasos lógicos serían:
* **Optimización de Hiperparámetros:** Utilizar herramientas como Optuna para afinar los parámetros del modelo LightGBM.
* **Ensamblado de Modelos:** Combinar las predicciones del modelo LightGBM y el modelo LSTM para crear una predicción final aún más robusta.
