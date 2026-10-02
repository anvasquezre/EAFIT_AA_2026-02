# Sesión 03 — Plan de cambios slide por slide

Plan de revisión de `03_ml_temporal_prediction.tex` a partir de:

- **`syllabusML.pdf`** → **syllabus oficial vigente** (6 sesiones de 4 h, 3 unidades). Es la referencia de contenido de este plan.
- **`Comentarios.docx`** → no tiene comentarios específicos de la sesión 3. El comentario general sobre métricas de regresión ya está cubierto en este deck (MAE, RMSE, MASE).
- **`notebooks/03_ml_time_series_walkforward.ipynb`** → notebook de la sesión (renombrado con prefijo 03_). En la sección 6 se alinea sección por sección con el deck.

> **Convención de numeración.** Los números de slide corresponden a la página del PDF compilado (`lectures/03_ml_temporal_prediction.pdf`, 119 páginas). Los marcadores `% ==== pN` del `.tex` señalan el inicio de cada sección. Las slides nuevas usan códigos (A1, W1, G1, …).
>
> **Acciones:** `MANTENER` · `EDITAR` · `MOVER` · `FUSIONAR` · `ELIMINAR` · `NUEVO` · `OPCIONAL`
>
> **Objetivo de tamaño:** pasar de **119 a ~60 slides** (60 núcleo + 2 opcionales). Hoy el deck tiene 17 separadores de sección, ~27 slides que son solo imagen y ~12 slides de práctica.

## 0. Estado: plan aplicado en una versión v2

| Archivo | Contenido |
|---|---|
| `03_ml_temporal_prediction.tex` | Deck original (119 slides), con autor y branding EAFIT actualizados. Se conserva sin cambios de contenido. |
| `03_ml_temporal_prediction_v2.tex` | **Plan de este README aplicado:** 62 slides (60 núcleo + 2 opcionales), mismas 11 secciones de la sección 3, sin desbordes. |

- La v2 reutiliza el preámbulo y las figuras del original. Solo incrusta 3 figuras (descomposición, walk-forward, error por horizonte) dentro de slides con texto.
- **Pendiente:** el enlace a la sección de modelo global del notebook aparece como `\pend{por publicar}` en el hands-on (H1) hasta que se agregue esa sección a `03_ml_time_series_walkforward.ipynb`.

---

## 1. Qué exige el syllabus real para la sesión 3

**Unidad 2 — Sequence, Time & Personalization (sesiones 3–4).**

Objetivos de la unidad relevantes para la S3:

- construir **modelos de forecasting globales** con features temporales;
- aplicar **walk-forward validation** y entender el **leakage** en series de tiempo;
- (S4) usar SHAP para interpretar modelos en contextos secuenciales.

| Bloque | Conceptos del syllabus (S3) |
|---|---|
| Modelos | **Global forecasting models** |
| Features | Lag features, rolling windows, **seasonal encoding** · **Windowing for ML models** · **Cyclical encodings** for time features |
| Validación | **Walk-forward validation, embargo periods** |
| Métricas | **MASE and alternative forecasting metrics** |
| Hands-on | (1) construir y evaluar un **modelo global** · (2) implementar **walk-forward** · (3) **comparar horizontes** y errores |

**Conexión con el resto del curso.** En la S1 se presentó `TimeSeriesSplit` como CV temporal ("si las features tienen cronología"). En la S2, Optuna exige elegir el tipo de fold según los datos (estratificado o temporal). La S3 desarrolla ese caso.

---

## 2. Brechas del deck frente al syllabus

### 2.1 Cobertura de conceptos

| Concepto del syllabus | Estado en el deck actual | Acción |
|---|---|---|
| Formulación temporal ($t$, $h$, $X_t$) | ✅ Slides 21–25 | Mantener, condensado |
| Lags y rolling windows | ✅ Slides 34–37 | Fusionar pares |
| **Seasonal encoding** | ⚠️ Solo seno/coseno de la hora (38–41) y seasonal naïve (46) | Añadir **términos de Fourier** y **lags estacionales** $y_{t-s}$ (F1) |
| **Windowing for ML models** | ⚠️ Fila supervisada y alineación (28–32), sin ventana deslizante ni estrategias multi-paso | **Nuevo:** W1 (ventana deslizante) y W2 (recursiva, directa, multi-output) |
| Cyclical encodings | ✅ Slides 38–41 | Fusionar |
| **Global forecasting models** | ❌ Solo una imagen con texto (94): *"la implementación multiserie completa queda fuera del núcleo"* | **Nueva sección** (G0–G3) con un dataset multiserie |
| Walk-forward validation | ✅ Slides 55–62 | Condensar |
| Embargo periods | ✅ Slide 61 | Mantener y añadir `TimeSeriesSplit(gap=…)` |
| MASE | ✅ Slides 73–74 | Fusionar |
| **Métricas alternativas** | ⚠️ MAE, RMSE, MAPE y sMAPE (71, 75); faltan las de **muchas series** | **Nuevo:** M1 (WAPE, RMSSE, MASE promediado por serie) |
| Hands-on: modelo global | ❌ No hay slide ni notebook | **H1** + sección nueva en el notebook |
| Hands-on: walk-forward | ✅ Notebook §16 | Referenciar en H1 |
| Hands-on: comparar horizontes | ✅ Slides 77–82, notebook §18 | Referenciar en H1 |

### 2.2 Contenido fuera del syllabus

| Contenido | Slides | Acción |
|---|---|---|
| Mapa de familias ARIMA / SARIMAX / redes temporales | 87–95 | Condensar en **O1** (`OPCIONAL`) |
| Estacionariedad | 18 | Condensar en **O2** (`OPCIONAL`) junto con la diferenciación |
| Diferenciación | 113 | **O2** (`OPCIONAL`) |

### 2.3 Inconsistencias detectadas

- ✅ **Autor y branding (aplicado):** portada, footer y slide 119 pasan de *Marco Teran* a *Andrés Vásquez Restrepo* (`avasquezr3@eafit.edu.co`). Se quitaron el footer dorado sobre gris y el estilo propio del índice: el deck usa el branding EAFIT de `eafitml.sty`, igual que 01a y 01b.
- ✅ **Fecha (aplicado):** "2026-2 – Medellín"; subtítulo "Aprendizaje Automático · Sesión 3".
- ✅ **Tipografía (aplicado):** la fuente EAFIT (Inter) es más ancha que Latin Modern y hacía desbordar 14 slides. La clase pasó de 12 pt a 11 pt y se ajustó la slide 107; el chequeo de desbordes queda limpio.
- **Datasets (slide 26):** lista Metro, Bike Sharing y otros. Hay que redefinir los roles: Metro para la metodología, **Electricity para el modelo global**, Bike solo como respaldo.
- **Notebook:** las slides 97 y 100 hablan de "NB03" sin enlace. Hay que enlazar `notebooks/03_ml_time_series_walkforward.ipynb` (repo `anvasquezre/EAFIT_AA_2026-02`, rama `main`) con `\pend{}` mientras no esté publicado.
- **Imágenes:** ~27 slides son capturas sin texto (slide raster completa). Se eliminan casi todas; solo se incrustan 3 figuras con valor didáctico (14, 57 y 79) dentro de slides con texto.

---

## 3. Nueva estructura de la sesión 3

```
A. Apertura                                   (1–2 + A1)            3
1. El tiempo cambia el experimento            (3–9)                 3
2. Leer la estructura temporal                (10–19)               5
3. Formulación: target y horizonte            (20–26)               4
4. Windowing: de serie a tabla supervisada    (27–32 + W1–W2)       6
5. Features: lags, rolling y calendario       (33–42 + F1)          5
6. Modelos globales                           (NUEVO: G0–G3)        4
7. Baselines temporales                       (43–48)               3
8. Validación temporal y leakage              (49–68)               9
9. Métricas y multi-horizonte                 (69–82 + M1)          7
10. Modelos, notebook y demo                  (83–104 + O1 + H1)    6
11. Cierre                                    (105–119 + O2)        7
                                                          Total    62  (60 núcleo + 2 opcionales)
```

El hilo narrativo queda así: **el tiempo rompe la intercambiabilidad → leer la serie → formular (t, h) → convertir la serie en ventanas → features válidas → de una serie a muchas (modelo global) → baselines honestos → validar hacia adelante sin leakage → medir contra referencias y por horizonte → entrenar y defender.**

El modelo global va **después** de las features (porque reutiliza lags y rolling por serie) y **antes** de baselines y validación, para que el walk-forward y las métricas de las secciones 8–9 se apliquen a una sola serie (Metro) y a muchas (Electricity).

> ⚠️ **Tiempo (4 h).** Unas 50 slides de teoría a ~3 min cada una (≈ 2,5 h) más ~1–1,5 h de hands-on con el notebook. Las slides O1 y O2 solo se presentan si sobra tiempo.

---

## 4. Cambios slide por slide

### 4.A Apertura (slides 1–2 + A1)

| Slide | Título actual | Comentario | Acción | Cambio propuesto |
|---|---|---|---|---|
| 1 | Portada | — | ✅ EDITAR | **Aplicado:** autor, fecha "2026-2 — Medellín" y subtítulo "Sesión 3". |
| 2 | Contenido | — | EDITAR | Se regenera sola con la nueva estructura de `\section`. |
| **A1** | — | Continuidad del curso | NUEVO | **Dónde quedamos + mapa de la sesión.** S1: `TimeSeriesSplit` como CV temporal. S2: el fold de Optuna depende de los datos. Hoy: una serie (Metro) y muchas series (Electricity). Lista de las 11 secciones. |

### 4.1 El tiempo cambia el experimento (slides 3–9)

| Slide | Título actual | Comentario | Acción | Cambio propuesto |
|---|---|---|---|---|
| 3 | Sección | — | MANTENER | — |
| 4 | La pregunta central de esta sesión | — | EDITAR | Absorbe el mensaje de la 6 (lo aprendido se vuelve más exigente) y la fórmula de la 7. |
| 5 | Imagen | — | ELIMINAR | Decorativa. |
| 6 | Lo aprendido antes se vuelve más exigente | — | FUSIONAR → 4 | — |
| 7 | El tiempo rompe la intercambiabilidad | — | FUSIONAR → 4 | Conservar $\hat y_{t+h} = f(X_t)$ con $X_t$ disponible solo hasta $t$. |
| 8 | Imagen | — | ELIMINAR | — |
| 9 | Chequeo inicial: ¿qué podía saber el modelo? | — | MANTENER | Buena apertura participativa. |

### 4.2 Leer la estructura temporal (slides 10–19)

| Slide | Título actual | Comentario | Acción | Cambio propuesto |
|---|---|---|---|---|
| 10 | Sección | — | MANTENER | — |
| 11 | Una serie de tiempo no es solo una fecha | — | EDITAR | Absorbe la 13 (componentes $y_t = T_t + S_t + R_t$) e incrusta la figura de la 14. |
| 12 | Imagen | — | ELIMINAR | — |
| 13 | Componentes básicos de una serie | — | FUSIONAR → 11 | — |
| 14 | Figura de descomposición | — | FUSIONAR → 11 | Se incrusta como figura. |
| 15 | Frecuencia y granularidad | — | MANTENER | — |
| 16 | Autocorrelación: memoria estadística | — | MANTENER | Añadir: "$\rho_s$ alto (por ejemplo $s=24$) justifica lags estacionales" (puente a F1). |
| 17 | Figura ACF | — | ELIMINAR | Opcional: incrustarla pequeña en la 16. |
| 18 | Estabilidad temporal / estacionariedad | Fuera del syllabus | MOVER → O2 | — |
| 19 | Checklist de diagnóstico antes de modelar | — | MANTENER | — |

### 4.3 Formulación: target y horizonte (slides 20–26)

| Slide | Título actual | Comentario | Acción | Cambio propuesto |
|---|---|---|---|---|
| 20 | Sección | — | MANTENER | — |
| 21 | "Predecir tráfico" no es una formulación completa | — | EDITAR | Absorbe la 22: contrato mínimo + definición de $t$, $h$, $X_t$, $y_{t+h}$. |
| 22 | La predicción temporal como problema supervisado | — | FUSIONAR → 21 | — |
| 23 | Imagen | — | ELIMINAR | — |
| 24 | El horizonte cambia la formulación | — | EDITAR | Añadir como pregunta final la práctica de la 25 (dos de sus frases vagas). |
| 25 | Práctica: complete la formulación temporal | — | FUSIONAR → 24 | — |
| 26 | El caso aplicado debe servir al método | Datasets | EDITAR | Nuevos roles: **Metro** = metodología (una serie); **Electricity** = modelo global (muchas series); Bike = respaldo. |

### 4.4 Windowing: de serie a tabla supervisada (slides 27–32 + W1–W2)

| Slide | Título actual | Comentario | Acción | Cambio propuesto |
|---|---|---|---|---|
| 27 | Sección: De serie temporal a tabla supervisada | Syllabus: *windowing* | EDITAR | "Windowing: de serie temporal a tabla supervisada". |
| 28 | Forecasting tabular: convertir memoria en features | — | EDITAR | Absorbe la 30 (la unidad real de entrenamiento: $z_t \to y_{t+h}$). |
| 29 | Imagen | — | ELIMINAR | — |
| 30 | La unidad real de entrenamiento | — | FUSIONAR → 28 | — |
| 31 | Ejemplo mínimo de alineación | — | MANTENER | Coincide con el notebook §6 (dummy manual). |
| 32 | Una tabla temporal puede estar mal aunque se vea completa | — | MANTENER | — |
| **W1** | — | Syllabus: *windowing for ML models* | NUEVO | **Ventana deslizante.** Cada fila = ventana de longitud $L$ (lookback) que mira hacia atrás y un objetivo $H$ pasos adelante. Diagrama TikZ de ventanas que avanzan sobre la serie. El tamaño de $L$ es un hiperparámetro y se elige con walk-forward. |
| **W2** | — | Syllabus: *windowing for ML models* | NUEVO | **Estrategias multi-paso.** **Recursiva** (un modelo a 1 paso que se realimenta; acumula error) · **directa** (un modelo por horizonte $h$, la que usa el notebook) · **multi-output** (un modelo predice $H$ salidas). Tabla: número de modelos, acumulación de error, costo y cuándo usar cada una. |

### 4.5 Features: lags, rolling y calendario (slides 33–42 + F1)

| Slide | Título actual | Comentario | Acción | Cambio propuesto |
|---|---|---|---|---|
| 33 | Sección | — | MANTENER | — |
| 34 | Lags y rolling windows convierten memoria en evidencia | — | EDITAR | Absorbe las fórmulas de la 36 ($\text{lag}_k(t)$, $\text{rolling\_mean}_w(t)$). |
| 35 | Imagen | — | ELIMINAR | — |
| 36 | Lags y ventanas: fórmulas | — | FUSIONAR → 34 | — |
| 37 | Feature temporal válida vs feature contaminada | — | EDITAR | Añadir como pregunta final la práctica de la 42 (clima observado frente a pronosticado). |
| 38 | El calendario es válido, pero debe codificarse como ciclo | — | EDITAR | Absorbe la 40 (seno/coseno de la hora). |
| 39 | Imagen | — | ELIMINAR | — |
| 40 | La hora no vive en una recta: vive en un ciclo | — | FUSIONAR → 38 | — |
| 41 + **F1** | Qué representa cada variable temporal | Syllabus: *seasonal encoding* | EDITAR | Añadir la **codificación estacional**: términos de Fourier $\sin(2\pi k t/s)$, $\cos(2\pi k t/s)$ con $k = 1..K$ para estacionalidades largas (semanal, anual) y **lags estacionales** $y_{t-s}$. Cuándo usar cada uno. |
| 42 | Práctica: calendario válido, variable externa dudosa | — | FUSIONAR → 37 | — |

### 4.6 NUEVO — Modelos globales (G0–G3)

| Slide | Acción | Contenido propuesto |
|---|---|---|
| **G0** | NUEVO | **Sección: De una serie a muchas: modelos globales.** |
| **G1** | NUEVO (absorbe la 94) | **Local frente a global.** Un modelo **local** aprende una serie; un modelo **global** aprende un solo $f$ con las filas de todas las series. **Gana:** más datos por modelo, patrones compartidos y series nuevas o cortas (cold start). **Arriesga:** series heterogéneas, escalas que dominan la pérdida, una serie grande que "manda". Ejemplo: 20 clientes de electricidad. |
| **G2** | NUEVO | **Cómo se arma la tabla global.** Apilar las series con `series_id` · lags y rolling **por serie** (`groupby(series_id).shift`) · **escalar por serie** (dividir por la media histórica de cada una, calculada solo con train) · features estáticas de la serie (tipo de cliente, región) · la identidad de la serie, como categórica o con target encoding. Diagrama: N series → una tabla larga. |
| **G3** | NUEVO | **Validar un modelo global.** Walk-forward con el **mismo corte temporal** para todas las series (nunca usar el futuro de una serie para predecir el pasado de otra) · embargo igual · métricas **por serie** y agregadas (puente a M1) · comparar contra modelos locales y contra seasonal naïve por serie. |

### 4.7 Baselines temporales (slides 43–48)

| Slide | Título actual | Comentario | Acción | Cambio propuesto |
|---|---|---|---|---|
| 43 | Sección | — | MANTENER | — |
| 44 | Un baseline temporal no es un modelo ridículo | — | EDITAR | Absorbe las fórmulas de la 46 (naïve, seasonal naïve, moving average). |
| 45 | Imagen | — | ELIMINAR | — |
| 46 | Tres referencias temporales obligatorias | — | FUSIONAR → 44 | — |
| 47 | Cuándo cada baseline pone presión real | — | EDITAR | Añadir la práctica de la 48 como pregunta final y la nota "en un modelo global, el baseline se calcula **por serie**". |
| 48 | Práctica: elija un baseline temporal | — | FUSIONAR → 47 | — |

### 4.8 Validación temporal y leakage (slides 49–68)

| Slide | Título actual | Comentario | Acción | Cambio propuesto |
|---|---|---|---|---|
| 49 | Sección: Random split | — | EDITAR | "Validación temporal: random split, walk-forward y leakage" (una sola sección para 49–68). |
| 50 | Random split puede ser una trampa temporal | — | EDITAR | Absorbe la 52 (definición formal de contaminación) y la 53 (tabla random / holdout / walk-forward). |
| 51 | Imagen | — | ELIMINAR | — |
| 52 | Qué significa contaminación temporal | — | FUSIONAR → 50 | — |
| 53 | Tres formas de separar datos temporales | — | FUSIONAR → 50 | — |
| 54 | Práctica: diagnostique el split | — | ELIMINAR | La cubre la 62. |
| 55 | Subsección: walk-forward | — | ELIMINAR | Separador innecesario. |
| 56 | Validar hacia adelante | — | EDITAR | Incrustar la figura de la 57. |
| 57 | Figura de walk-forward | — | FUSIONAR → 56 | — |
| 58 | La restricción cronológica del fold | — | EDITAR | Absorbe la 59 (score final $\hat E_{WF}$ como promedio de folds). |
| 59 | El score final resume varios futuros evaluados | — | FUSIONAR → 58 | — |
| 60 | Variantes útiles de validación temporal | — | MANTENER | Holdout, ventana expansiva, ventana deslizante. |
| 61 | Embargo | Syllabus: *embargo periods* | EDITAR | Añadir el código mínimo `TimeSeriesSplit(n_splits=5, test_size=…, gap=24)` y una línea que conecte con la CV temporal de la S1. |
| 62 | Práctica: diseñe el protocolo | — | EDITAR | Añadir la variante global: "¿cómo cambia el protocolo si son 20 clientes?". |
| 63 | Subsección: leakage | — | ELIMINAR | — |
| 64 | Leakage temporal no siempre parece leakage | — | EDITAR | Absorbe la 67 (regla formal $x_j(t) \in I_t$). |
| 65 | Imagen | — | ELIMINAR | — |
| 66 | Cinco rutas comunes de leakage temporal | — | EDITAR | Añadir la ruta del modelo global (**escalar por serie con toda la historia**) y la práctica de la 68 como pregunta final. |
| 67 | La regla formal: disponibilidad antes de $t$ | — | FUSIONAR → 64 | — |
| 68 | Práctica: audite el pipeline temporal | — | FUSIONAR → 66 | — |

### 4.9 Métricas y multi-horizonte (slides 69–82 + M1)

| Slide | Título actual | Comentario | Acción | Cambio propuesto |
|---|---|---|---|---|
| 69 | Sección | — | MANTENER | — |
| 70 | Un error bajo no siempre significa un modelo útil | — | FUSIONAR → 71 | Su mensaje pasa a subtítulo de la 71. |
| 71 | MAE y RMSE | — | EDITAR | Subtítulo: "el error se lee contra un baseline y un horizonte". |
| 72 | Imagen | — | ELIMINAR | — |
| 73 | MASE: escalar el error con una referencia temporal | Syllabus: *MASE* | EDITAR | Absorbe la tabla de interpretación de la 74 (MASE < 1, ≈ 1, > 1) y la variante estacional ($m = 24$). |
| 74 | Cómo interpretar MASE | — | FUSIONAR → 73 | — |
| 75 | MAPE y sMAPE | Syllabus: *alternative metrics* | EDITAR | Añadir **WAPE** $= \frac{\sum\lvert y-\hat y\rvert}{\sum\lvert y\rvert}$ como alternativa estable con ceros, y la práctica de la 76 como pregunta. |
| 76 | Práctica: ¿qué métrica usaría? | — | FUSIONAR → 75 | — |
| **M1** | — | Syllabus: *alternative metrics* + modelo global | NUEVO | **Métricas para muchas series.** Por qué no promediar MAE entre series de escalas distintas. **MASE promedio por serie**, **WAPE agregado**, **RMSSE** (raíz del error cuadrático escalado; usado en M5). Tabla: qué pregunta responde cada una. |
| 77 | Subsección: multi-horizon | — | ELIMINAR | — |
| 78 | Un modelo no tiene un único desempeño temporal | — | EDITAR | Absorbe la 80 ($E_h$ por horizonte) e incrusta la figura de la 79. |
| 79 | Figura error por horizonte | — | FUSIONAR → 78 | — |
| 80 | Evaluar por horizonte | — | FUSIONAR → 78 | — |
| 81 | Cómo cambia la utilidad por horizonte | — | EDITAR | Añadir la práctica de la 82 (modelo A frente a B) como pregunta final. |
| 82 | Práctica: lea resultados por horizonte | — | FUSIONAR → 81 | — |

### 4.10 Modelos, notebook y demo (slides 83–104 + O1 + H1)

| Slide | Título actual | Comentario | Acción | Cambio propuesto |
|---|---|---|---|---|
| 83 | Sección | — | MANTENER | — |
| 84 | Primero formulación, después complejidad | — | EDITAR | Absorbe la 85. Modelos: Ridge y HGBR para Metro; **HGBR o LightGBM global** para Electricity, todos bajo el mismo protocolo. |
| 85 | Dos modelos canónicos | — | FUSIONAR → 84 | — |
| 86 | Imagen | — | ELIMINAR | — |
| 87 | Subsección: mapa de modelos temporales | Fuera del syllabus | ELIMINAR | — |
| 88 | Imagen | — | ELIMINAR | — |
| 89 | Tres familias, una misma regla de credibilidad | Fuera del syllabus | FUSIONAR → **O1** | — |
| 90 | Imagen | — | ELIMINAR | — |
| 91 | SARIMAX agrega covariables | Fuera del syllabus | FUSIONAR → **O1** | Conservar la regla "exógena futura no disponible = leakage". |
| 92–93 | Imágenes | — | ELIMINAR | — |
| 94 | Un modelo local aprende una sola serie… | Syllabus: *global models* | MOVER → **G1** | — |
| 95 | Imagen | — | ELIMINAR | — |
| **O1** | — | Fuera del syllabus | OPCIONAL | **Familias de modelos temporales y una misma regla de credibilidad:** ARIMA/SARIMA(X), ML tabular y redes temporales compiten bajo el mismo target, horizonte, baseline y validación. Una tabla; la implementación queda en el notebook (§20). |
| 96 | Subsección: del criterio al experimento | — | ELIMINAR | — |
| 97 | El notebook ejecuta el contrato metodológico | — | FUSIONAR → 99 | Su mensaje pasa a subtítulo. |
| 98 | Imagen | — | ELIMINAR | — |
| 99 | Pseudocódigo: target, features y validación | — | EDITAR | Absorbe la tabla de la 100. Añadir el paso global: "lags con `groupby(series_id)`; escala por serie ajustada en train". |
| 100 | Qué debe inspeccionarse en NB03 | — | FUSIONAR → 99 | Añadir la fila "modelo global frente a local: ¿mejora MASE y WAPE por serie?". |
| 101 | Subsección: demo Metro | — | ELIMINAR | — |
| 102 | La historia aplicada de la demo | — | EDITAR | Absorbe la 104 (qué debe demostrar la demo). |
| 103 | Imagen | — | ELIMINAR | — |
| 104 | Qué debe demostrar la demo | — | FUSIONAR → 102 | — |
| **H1** | — | Syllabus: hands-on | NUEVO | **Hands-on de la sesión** con enlace al notebook (`\pend{}` mientras no esté publicado): (1) **modelo global** en Electricity frente a modelos locales y seasonal naïve; (2) **walk-forward** con embargo en Metro; (3) **comparar horizontes** {1, 3, 6, 24} con MAE y MASE. |

### 4.11 Cierre (slides 105–119 + O2)

| Slide | Título actual | Comentario | Acción | Cambio propuesto |
|---|---|---|---|---|
| 105 | Sección | — | MANTENER | — |
| 106 | Imagen | — | ELIMINAR | — |
| 107 | Un resultado temporal defendible es una cadena | — | EDITAR | Absorbe las cinco preguntas de la 111 y el mensaje de la 117. |
| 108 | Sección: Apéndice | — | ELIMINAR | — |
| 109 | Glosario | — | EDITAR | Añadir: modelo global y local, ventana deslizante, estrategia directa y recursiva, términos de Fourier, WAPE, RMSSE, embargo (`gap`). |
| 110 | Imagen | — | ELIMINAR | — |
| 111 | Diagnóstico rápido de invalidez | — | FUSIONAR → 107 | — |
| 112 | Imagen | — | ELIMINAR | — |
| 113 | Diferenciación como nota opcional | Fuera del syllabus | FUSIONAR → **O2** | — |
| **O2** | — | Fuera del syllabus | OPCIONAL | **Estabilidad y diferenciación:** estacionariedad como criterio práctico (de la 18) + diferencia regular y estacional ($\nabla y_t$, $\nabla_s y_t$) (de la 113). |
| 114 | Mini-casos para cierre o discusión | — | ELIMINAR | Las prácticas fusionadas en cada sección ya cubren la discusión. Pueden pasar a la guía del taller. |
| 115 | Cómo esta sesión alimenta el taller modular | — | FUSIONAR → 116 | — |
| 116 | Preguntas de revisión para examen y PI | — | EDITAR | Añadir: "¿Cuándo usaría un modelo global en lugar de uno por serie?", "¿Directa o recursiva para 24 pasos?", "¿Por qué no promediar MAE entre series?", "¿Qué hace el `gap` del walk-forward?". |
| 117 | Lo que debe quedar claro | — | FUSIONAR → 107 | — |
| 118 | Transición a la siguiente sesión | — | MANTENER | Recomendadores + SHAP (S4). |
| 119 | Gracias por su atención | — | ✅ EDITAR | **Aplicado:** igual que en 01a/01b ("Aprendizaje Automático · SI7009 · Universidad EAFIT"). |

---

## 5. Trazabilidad: syllabus de la S3 → slides

| Concepto del syllabus | Slides (plan) |
|---|---|
| Global forecasting models | **G0–G3**, 26, 84, 99 |
| Lag features, rolling windows | 34, 37 |
| Seasonal encoding | 16, **41 + F1**, 44 (seasonal naïve) |
| Windowing for ML models | 28, 31, 32, **W1–W2** |
| Cyclical encodings | 38 |
| Walk-forward validation | 50, 56, 58, 60, 62, **G3** |
| Embargo periods | 61 |
| MASE | 73 |
| Alternative forecasting metrics | 71, 75, **M1** |
| Hands-on (1) modelo global | **H1** + notebook (sección nueva) |
| Hands-on (2) walk-forward | **H1** + notebook §16 |
| Hands-on (3) comparar horizontes | 78, 81, **H1** + notebook §18 |

---

## 6. Alineación con el notebook `03_ml_time_series_walkforward.ipynb`

| Sección del notebook | Slides del nuevo deck | Estado |
|---|---|---|
| §1–2 Contrato y conceptos previos | A1, sección 1 | ✅ |
| §3–5 Setup y funciones auxiliares (`mase`, `add_lag_features`, `add_rolling_features`, `add_cyclical_features`, `make_supervised_forecasting_frame`) | — | ✅ (soporte) |
| §6 Dummy manual: alineación lag ↔ target | 31 | ✅ |
| §7 Serie sintética: tendencia, estacionalidad y ruido | 11 | ✅ |
| §8 Codificación cíclica | 38, 41 | ✅ · falta Fourier (F1) |
| §9–10 Dataset Metro/Bike y auditoría inicial | 19, 26 | ✅ |
| §11 Serie → tabla supervisada (`add_forecast_target` por horizonte) | sección 4 | ✅ · ya usa la estrategia **directa** sin nombrarla (W2) |
| §12 Split temporal train/val/test | sección 8 | ✅ |
| §13 Baselines (naïve, seasonal naïve, MA) | sección 7 | ✅ |
| §14–15 Ridge, HGBR y holdout temporal | 84 | ✅ |
| §16 Walk-forward validation | 56, 58, 60 | ⚠️ sin **embargo / `gap`** (61) |
| §17 Random split como anti-ejemplo | 50 | ✅ |
| §18 Multi-horizonte {1, 3, 6, 24} | 78, 81 | ✅ |
| §19 Test temporal reservado | 102 | ✅ |
| §20 Extensiones ARIMA/SARIMA/XGBoost/LightGBM | O1 | ✅ (opcional) |
| §21–25 Registro de evidencia, auditoría, ejercicios y cierre | 107, 116 | ✅ |
| — | **G0–G3, M1, H1 (ej. 1)** | ❌ **no hay modelo global** ni WAPE/RMSSE |

**Cambios propuestos al notebook** (pendientes, no aplicados):

- [ ] **Sección nueva "Modelo global frente a local"** con **UCI Electricity Load Diagrams 2011–2014** (dataset UCI 321: 370 clientes, consumo cada 15 min).
  - Subconjunto: ~20 clientes, re-muestreados a frecuencia horaria, 2013–2014.
  - Contenido: `series_id`, lags y rolling con `groupby`, escala por serie ajustada en train, un HGBR o LightGBM **global** frente a uno **por serie** y a seasonal naïve, walk-forward con corte común y métricas por serie (MASE promedio, WAPE).
  - Verificar la URL de descarga directa de UCI y el tamaño (~250 MB comprimido); considerar guardar el subconjunto como CSV pequeño.
- [ ] **Embargo** en el walk-forward existente (§16): parámetro `gap`, conectado con la slide 61.
- [ ] **Términos de Fourier** junto a la codificación cíclica (§8), conectado con F1.
- [ ] Una celda que nombre la **estrategia directa** (§11) y contraste con la recursiva (W2).
- [ ] **WAPE y RMSSE** en `evaluate_regression_predictions` (§5), conectado con 75 y M1.
- [ ] Badges de Colab y Kaggle hacia `anvasquezre/EAFIT_AA_2026-02` (rama `main`), como en los notebooks 01 y 02, y celda de instalación para Colab.

---

## 7. Pendientes para el docente

- [x] Autor, correo y branding EAFIT (portada, footer, índice y slide 119).
- [x] Fecha "2026-2" en la portada.
- [ ] Confirmar **UCI Electricity** como dataset del modelo global (y la URL).
- [ ] Enlace del notebook en H1 (`\pend{}` hasta publicarlo).
- [ ] Diagramas nuevos en TikZ: ventana deslizante (W1) y apilado de series (G2).
- [ ] Decidir si O1 y O2 se presentan o quedan como lectura.

## 8. Resumen de conteo

| Tipo de cambio | Slides |
|---|---|
| Nuevas núcleo (A1, W1, W2, G0–G3, M1, H1) | 9 |
| Nuevas opcionales (O1, O2) | 2 |
| Mantenidas o editadas (con contenido fusionado) | 51 |
| Fusionadas en otra slide | 34 |
| Eliminadas (imágenes, separadores y prácticas cubiertas) | 32 |
| Movidas (18 → O2, 94 → G1) | 2 |
| **Total del deck** | **60 núcleo + 2 opcionales = 62** (antes 119) |
