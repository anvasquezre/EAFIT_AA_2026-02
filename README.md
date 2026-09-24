[![banner](/_assets/pics/bannerAI.jpg)](https://github.com/marcoteran/ml)

# SI7009 · Aprendizaje Automático

### Maestría en Ciencia de Datos y Analítica · Universidad EAFIT

<p align="left">
  <img src="https://img.shields.io/badge/curso-SI7009-0F2747?style=for-the-badge" alt="SI7009">
  <img src="https://img.shields.io/badge/programa-Maestría%20en%20Ciencia%20de%20Datos-173B63?style=for-the-badge" alt="Maestría">
  <img src="https://img.shields.io/badge/modalidad-Intensivo-1F2937?style=for-the-badge" alt="Intensivo">
  <img src="https://img.shields.io/badge/enfoque-Aplicado%20%2B%20Reproducible-6B7280?style=for-the-badge" alt="Aplicado y reproducible">
</p>

Repositorio oficial del curso **Aprendizaje Automático**.

Este curso desarrolla criterio técnico para formular, validar, comparar e interpretar modelos de **Machine Learning** en escenarios reales: datos desbalanceados, validación temporal, selección de métricas, boosting moderno, recomendación, clustering, interpretabilidad y confiabilidad del modelo.

**Profesor:** [Marco Teran](https://marcoteran.github.io/)

---

## Acceso rápido

- [Ruta del curso, materiales y recursos por sesión](#ruta-del-curso-materiales-y-recursos-por-sesión)
- [Evaluación](#evaluación)
- [Taller evaluativo](#taller-evaluativo)
- [Exposiciones](#exposiciones)
- [Datasets principales](#datasets-principales)
- [Recursos recomendados](#recursos-recomendados)
- [Estructura del repositorio](#estructura-del-repositorio)

---

## Ruta del curso, materiales y recursos por sesión

| Sesión | Tema central | Problema guía | Materiales | Recursos asociados |
|---|---|---|---|---|
| 1 | Syllabus, evaluación moderna, desbalance y fundamentos de ensembles | Evaluación honesta de modelos | [Slides PDF](https://github.com/marcoteran/ml/raw/master/lectures/01_ml_modern_evaluation_imbalance_ensembles.pdf)<br>[Notebook NB01](https://github.com/marcoteran/ml/blob/master/notebooks/01_ml_flujo_basico_fraude.ipynb)<br>[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marcoteran/ml/blob/master/notebooks/01_ml_flujo_basico_fraude.ipynb)<br>[![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/marcoteran/ml/blob/master/notebooks/01_ml_flujo_basico_fraude.ipynb) | [Scikit-learn · Model selection](https://scikit-learn.org/stable/model_selection.html)<br>[Scikit-learn · Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)<br>[Scikit-learn · Threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html) |
| 2 | Boosting moderno, optimización de hiperparámetros, interpretabilidad y diagnóstico de modelos | Credit Card Fraud Detection | [Slides PDF](https://drive.google.com/file/d/1tmNx19iKF9JwbQAYxu0yuWfawEAIzcUz/view?usp=drive_link)<br>[Notebook NB02](https://github.com/marcoteran/ml/blob/master/notebooks/ml_boosting_optuna.ipynb)<br>[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marcoteran/ml/blob/master/notebooks/ml_boosting_optuna.ipynb)<br>[![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/marcoteran/ml/blob/master/notebooks/ml_boosting_optuna.ipynb) | [XGBoost](https://xgboost.readthedocs.io/)<br>[XGBoost · Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)<br>[LightGBM](https://lightgbm.readthedocs.io/en/stable/)<br>[CatBoost](https://catboost.ai/)<br>[Optuna](https://optuna.org/)<br>[SHAP](https://shap.readthedocs.io/en/latest/)<br>[SHAP · Introduction to Shapley values](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html)<br>[MLflow](https://mlflow.org/)<br>[Residual plots](https://statisticsbyjim.com/regression/check-residual-plots-regression-analysis/)<br>[Applied ML article for critical metric discussion](https://www.mdpi.com/2075-4418/14/24/2813) |
| 3 | Series de tiempo, forecasting tabular y walk-forward validation | Predicción temporal con datos tabulares | [Slides PDF](https://github.com/marcoteran/ml/raw/master/lectures/03_ml_temporal_prediction.pdf)<br>[Notebook NB02](https://github.com/marcoteran/ml/blob/master/notebooks/ml_time_series_walkforward.ipynb)<br>[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marcoteran/ml/blob/master/notebooks/ml_time_series_walkforward.ipynb)<br>[![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/marcoteran/ml/blob/master/notebooks/ml_time_series_walkforward.ipynb) | [Scikit-learn · Time-related feature engineering](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html)<br>[Forecasting: Principles and Practice](https://otexts.com/fpp3/)<br>[Skforecast](https://skforecast.org/)<br>[Respect the order: cross-validation in time series](https://medium.com/@pacosun/respect-the-order-cross-validation-in-time-series-7d12beab79a1) |
| 4 | Representación, clustering aplicado y confiabilidad del modelo | Online Retail | [Slides PDF](https://github.com/marcoteran/ml/raw/master/lectures/04_ml_unsupervisedlearning.pdf)<br>[Notebook NB02](https://github.com/marcoteran/ml/blob/master/notebooks/ml_unsupervisedlearning.ipynb)<br>[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marcoteran/ml/blob/master/notebooks/ml_unsupervisedlearning.ipynb)<br>[![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/marcoteran/ml/blob/master/notebooks/ml_unsupervisedlearning.ipynb) | [Scikit-learn · Clustering](https://scikit-learn.org/stable/modules/clustering.html)<br>[Scikit-learn · Novelty and outlier detection](https://scikit-learn.org/stable/modules/outlier_detection.html)<br>[Scikit-learn · Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)<br>[UCI · Online Retail](https://archive.ics.uci.edu/dataset/352/online%2Bretail) |

---

## Evaluación

| Componente | Peso |
|---|---:|
| Taller modular único | 20% |
| Exposición de temas avanzados | 20% |
| Examen final | 25% |
| Proyecto Integrador | 35% |

### Examen final

- **Formato:** teórico, opción múltiple
- **Preguntas:** 20
- **Plataforma:** Interactiva
- **Duración y ventana:** según configuración oficial en Interactiva

El examen evalúa comprensión conceptual, lectura crítica de resultados, selección de métricas, validación, comparación de modelos, interpretabilidad y toma de decisiones metodológicas.

---

## Taller evaluativo

El taller corresponde a una evaluación grupal del curso y se entrega por Interactiva.

Consiste en desarrollar un flujo completo de **Machine Learning para predicción binaria desbalanceada**, usando una base de datos en SQLite para estimar la ocurrencia de accidentes por barrio y hora.

**Entregables:**

- informe en PDF;
- notebook `.ipynb` ejecutable;
- resultados reproducibles;
- recomendación técnica final.

**Descarga:** [Enunciado del taller](https://github.com/marcoteran/ml/raw/master/homeworks/aprendizaje_automatico_taller_20261.pdf)

El taller evalúa especialmente:

- formulación del problema;
- construcción del target;
- generación de casos negativos;
- análisis exploratorio y calidad de datos;
- unión de tablas relacionales;
- ingeniería de características;
- manejo de clases desbalanceadas;
- validación temporal;
- control de fuga de información;
- comparación de modelos;
- ajuste de hiperparámetros;
- selección de métricas;
- thresholding;
- matriz de confusión;
- propuesta de uso operativo;
- recomendación final defendible.

---

## Exposiciones

Cada grupo podrá escoger un tema avanzado de Machine Learning por orden de selección.

La presentación debe ser breve, clara y técnica. No se espera una revisión enciclopédica, sino una explicación bien estructurada del problema, la intuición del método, un ejemplo computacional y sus riesgos de uso.

### Entregables

- presentación en PDF;
- repositorio público en GitHub;
- ejemplo computacional reproducible en Python.

### Banco de temas

- Reinforcement Learning
- Federated Learning
- Semi-Supervised Learning
- MLOps / Deployment
- Explainable AI
- Conformal Prediction
- Uncertainty Quantification
- Active Learning
- Learning to Rank
- Graph Machine Learning
- Tabular Deep Learning
- Anomaly Detection at Scale
- Continual Learning
- AutoML
- Time-Series Transformers
- TinyML / Edge ML

---

## Datasets principales

| Dataset | Descripción | Uso dentro del curso |
|---|---|---|
| [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | Transacciones anonimizadas con una clase positiva muy poco frecuente. | Clasificación desbalanceada, métricas, thresholds, boosting, HPO e interpretabilidad. |
| [Credit Card Transactions Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) | Transacciones simuladas con variables crudas (monto, categoría, hora, ubicación, edad) y un periodo posterior para inferencia. | Flujo de ML, EDA, outliers, pipelines de inferencia, métricas, threshold y calibración (NB01). |
| [Metro Interstate Traffic Volume](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) | Datos horarios de tráfico con variables de clima, calendario y volumen vehicular. | Forecasting tabular, ingeniería de variables temporales y validación walk-forward. |
| [MovieLens](https://grouplens.org/datasets/movielens/) | Calificaciones de usuarios a películas, ampliamente usado en recommendation systems. | Collaborative filtering, matrices usuario-item, ranking top-N y evaluación de recomendadores. |
| [Online Retail](https://archive.ics.uci.edu/dataset/352/online%2Bretail) | Transacciones reales de comercio electrónico con clientes, productos, cantidades y fechas. | Segmentación, clustering, representación de clientes y análisis de comportamiento transaccional. |

---

## Recursos recomendados

Estos recursos complementan la ruta del curso. La prioridad sugerida es comenzar por documentación oficial, luego revisar papers y finalmente usar tutoriales como apoyo práctico.

<details>
<summary><strong>Fundamentos, evaluación y validación</strong></summary>

- [Scikit-learn · Model selection and evaluation](https://scikit-learn.org/stable/model_selection.html)
- [Scikit-learn · Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Scikit-learn · Tuning the decision threshold](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [Scikit-learn · Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [Google Developers · Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
- [Made With ML · Practical ML guides](https://madewithml.com/)

</details>

<details>
<summary><strong>Boosting moderno, HPO y experimentación</strong></summary>

- [XGBoost documentation](https://xgboost.readthedocs.io/)
- [XGBoost · Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [LightGBM documentation](https://lightgbm.readthedocs.io/en/stable/)
- [CatBoost documentation](https://catboost.ai/)
- [Optuna documentation](https://optuna.org/)
- [Optuna · Read the Docs](https://optuna.readthedocs.io/)
- [MLflow](https://mlflow.org/)
- [MLflow documentation](https://mlflow.org/docs/latest/ml/)
- [Tutorial complementario · Random Forest tuning with Optuna](https://medium.com/@ehsannabatchian/optimizing-random-forest-models-a-deep-dive-into-hyperparameter-tuning-with-optuna-b8e4fe7f3670)

</details>

<details>
<summary><strong>Papers y referencias técnicas</strong></summary>

- [XGBoost paper](https://arxiv.org/abs/1603.02754)
- [LightGBM paper](https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
- [CatBoost paper](https://arxiv.org/abs/1810.11363)
- [Applied ML article for critical discussion of reported metrics](https://www.mdpi.com/2075-4418/14/24/2813)

</details>

<details>
<summary><strong>Interpretabilidad, SHAP y diagnóstico de modelos</strong></summary>

- [SHAP documentation](https://shap.readthedocs.io/en/latest/)
- [SHAP · Introduction to explainable AI with Shapley values](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html)
- [DataCamp · Introduction to SHAP values](https://www.datacamp.com/tutorial/introduction-to-shap-values-machine-learning-interpretability)
- [Statistics by Jim · Residual plots for regression analysis](https://statisticsbyjim.com/regression/check-residual-plots-regression-analysis/)

</details>

<details>
<summary><strong>Series de tiempo y forecasting</strong></summary>

- [Scikit-learn · Time-related feature engineering](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html)
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/)
- [Skforecast documentation](https://skforecast.org/)
- [Respect the order: cross-validation in time series](https://medium.com/@pacosun/respect-the-order-cross-validation-in-time-series-7d12beab79a1)

</details>

<details>
<summary><strong>Recommendation systems</strong></summary>

- [GroupLens · MovieLens datasets](https://grouplens.org/datasets/movielens/)
- [Implicit library documentation](https://benfred.github.io/implicit/)
- [Dive into Deep Learning · MovieLens example](https://d2l.ai/chapter_recommender-systems/movielens.html)
- [A practical guide to building recommender systems](https://towardsdatascience.com/a-guide-to-recommender-systems-6c0f0f1a7e47)

</details>

<details>
<summary><strong>Clustering, representación y confiabilidad</strong></summary>

- [Scikit-learn · Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [Scikit-learn · Novelty and outlier detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
- [Scikit-learn · Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [UCI Machine Learning Repository · Online Retail](https://archive.ics.uci.edu/dataset/352/online%2Bretail)

</details>

<details>
<summary><strong>Videos y charlas</strong></summary>

- [TED · Jeremy Howard — The wonderful and terrifying implications of computers that can learn](https://www.ted.com/talks/jeremy_howard_the_wonderful_and_terrifying_implications_of_computers_that_can_learn)
- [StatQuest · ROC and AUC, Clearly Explained](https://www.youtube.com/watch?v=4jRBRDbJemM)
- [StatQuest · Precision, Recall, Sensitivity and Specificity](https://www.youtube.com/watch?v=vP06aMoz4v8)
- [StatQuest · XGBoost in Python from Start to Finish](https://www.youtube.com/watch?v=GrJP9FLV3FE)
- [StatQuest · XGBoost Part 1](https://www.youtube.com/watch?v=OtD8wVaFm6E)
- [StatQuest · XGBoost Part 2](https://www.youtube.com/watch?v=8b1JEDvenQU)
- [Stanford recommender systems lecture](https://www.youtube.com/watch?v=GIcuSNAAa4g)

</details>

<details>
<summary><strong>Libros recomendados</strong></summary>

- Bishop · *Pattern Recognition and Machine Learning*
- Murphy · *Machine Learning: A Probabilistic Perspective*
- Hastie, Tibshirani, Friedman · *The Elements of Statistical Learning*
- Géron · *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*
- Aggarwal · *Recommender Systems: The Textbook*

</details>

---

## Estructura del repositorio

```text
ml/
├── README.md
├── lectures/
├── notebooks/
├── homeworks/
├── projects/
├── aditionalmaterial/
└── _assets/
````

> Nota: si la carpeta `aditionalmaterial/` ya existe con ese nombre en el repositorio, conviene mantenerla para no romper rutas. Si aún no está consolidada, se recomienda renombrarla como `additionalmaterial/`.

---

## Nota final

Un buen modelo no es solo el que obtiene el mayor score.

Un buen modelo es aquel que está bien formulado, validado sin fuga de información, comparado contra baselines razonables, interpretado con prudencia, documentado de forma reproducible y conectado con una decisión real.