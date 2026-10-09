# Sesión 05 — Plan de contenido slide por slide (reducción de dimensionalidad)

Plan para **crear desde cero** las slides de la sesión 5. Se basa en:

- **`syllabusML.pdf`** → Unidad 3, sesión 5 (*Dimensionality Reduction & Clustering*). En este curso la sesión 5 se dedica a la reducción de dimensionalidad y la sesión 6 al clustering (ver `../06_ml_clustering_anomalies/README.md`).
- **Deck heredado** `../04_ml_unsupervisedlearning/04_ml_unsupervisedlearning.tex` → solo como referencia de tono ("lectura responsable" de embeddings). No se reutiliza su estructura.
- **Taller, Módulo C** (`homeworks/latex/ml_taller_20262.tex`) → pide PCA y UMAP o t-SNE coloreados por segmento. La sesión debe dejar a los estudiantes listos para eso sin resolverlo por ellos.

> **Convención.** Cada slide tiene un código (A1, R1, P1, …). Todas las filas son `NUEVO`.
>
> **Tamaño:** **48 slides** para **~1 h 30 min** (≈ 1,9 min por slide; los `\sectionframe` cuentan pero no consumen tiempo). Máximo permitido: 50.
>
> **Fecha de la clase:** viernes 09/10/2026, 5:00–9:00 p. m. (1 h 30 de clase; el resto del bloque son exposiciones).

---

## 0. Estado actual

| Pieza | Estado |
|---|---|
| Slides de la S5 | ✅ `05_ml_dimensionality_reduction.tex` → 49 páginas (48 del plan + «Gracias»). Compilar con `latexmk -pdf` (copia local de `eafitml.sty`). |
| Figuras | ✅ `../make_figures_s5_s6.py --db olist.duckdb` genera `figures/*.png` y `figures/results_s5.json` (números citados en las slides; reproducible). |
| Notebook | ⚪ Fuera de alcance por ahora (ver pendientes). |
| Branding | ✅ `../eafitml.sty`, igual que S1–S4. |

---

## 1. Qué exige el syllabus

**Unidad 3 — Structure, Unsupervised Learning, Clustering & Reliability.** Objetivo de la unidad que cubre esta sesión: *Apply PCA and nonlinear DR (t-SNE, UMAP)*.

| Bloque | Concepto del syllabus |
|---|---|
| Lineal | **PCA** |
| No lineal | **t-SNE**, **UMAP** |
| Hands-on | (1) aplicar PCA, t-SNE y UMAP sobre un dataset real · (2) **evaluar embeddings visual y numéricamente** |

### 1.1 Lo que el syllabus no nombra pero la sesión necesita

| Concepto | Por qué |
|---|---|
| Maldición de la dimensionalidad | Motiva toda la sesión: con $p$ grande las distancias se concentran y "vecino" pierde sentido. |
| Escalado y transformaciones | En no supervisado el escalado **es** el modelo: decide qué variable domina la distancia. |
| Trustworthiness y preservación de $k$-vecinos | El hands-on (2) pide evaluar embeddings **numéricamente**. Sin una métrica, solo queda la opinión sobre una figura. |
| Estabilidad entre semillas | t-SNE y UMAP son estocásticos; un patrón que no se repite no es evidencia. |

### 1.2 Conexión con el resto del curso

| Sesión | Concepto que reaparece |
|---|---|
| S1 | `fit` solo en train, `Pipeline`, leakage → PCA también se ajusta dentro del pipeline. |
| S2 | Número de componentes como hiperparámetro en CV. |
| S4 | Los embeddings de ALS ya eran una reducción de dimensión de la matriz de utilidad. |
| S6 | El espacio construido hoy es la entrada del clustering; el error de reconstrucción de PCA reaparece como score de anomalía. |

---

## 2. Decisiones de diseño

| Decisión | Propuesta | Motivo |
|---|---|---|
| **Dataset** | **Olist**, unidad de análisis = **vendedor** (`seller_id`, ~3 k filas). Features: ventas totales, nº de órdenes, ticket medio, flete relativo, % de órdenes con retraso, calificación media, nº de categorías, región (estado). Construidas con SQL sobre `olist.duckdb`. | Mismo dataset del taller (continuidad, misma base DuckDB). El taller pide **clientes**: en clase se enseña el método con otra unidad, así no se regala el Módulo C. ~3 k filas permite t-SNE y UMAP en segundos. |
| **Mini-mundo** | 6 puntos en 2D: A(1,2), B(2,1), C(2,3), D(6,5), E(7,7), F(8,6). Se reutiliza en la S6 (K-Means, silueta, DBSCAN). | Permite calcular PCA a mano con números verificables (ver P3). |
| **Datos sintéticos** | `sklearn.datasets`: swiss roll, blobs en alta dimensión. | Muestran fallos y la maldición de la dimensionalidad con control total. |
| **Librerías** | `scikit-learn` (`PCA`, `TSNE`, `trustworthiness`), `umap-learn`. | `umap-learn` ya está en `pyproject.toml`. |

### 2.1 Números del mini-mundo (verificados con NumPy)

| Paso | Valor |
|---|---|
| Media | $\bar x = (4{,}33,\ 4{,}00)$ |
| Covarianza | $\Sigma = \begin{pmatrix} 9{,}07 & 6{,}60 \\ 6{,}60 & 5{,}60\end{pmatrix}$ |
| Eigenvalores | $\lambda_1 = 14{,}16$, $\lambda_2 = 0{,}51$ |
| PC1 | $\mathbf{w}_1 \approx (0{,}79,\ 0{,}61)$ (el signo es arbitrario) |
| Varianza explicada | PC1 = **96,5 %**, PC2 = 3,5 % |
| $z_1$ (coordenada en PC1) | A −3,86 · B −3,68 · C −2,46 · D 1,93 · E 3,94 · F 4,12 |

---

## 3. Estructura de la sesión 5

```
A. Apertura                                    A1–A3    3
1. Por qué reducir dimensión                   R0–R5    6
2. PCA                                         P0–P10  11
3. Más allá de lo lineal                       M0–M2    3
4. t-SNE                                       T0–T6    7
5. UMAP                                        U0–U5    6
6. Evaluar un embedding                        E0–E5    6
7. Hands-on Olist y cierre                     Z1–Z6    6
                                               Total   48
```

**Hilo narrativo:** hasta la S4 siempre hubo un $y$. Hoy no hay. Una fila es un punto en $\mathbb{R}^p$ → con $p$ grande la geometría engaña → PCA comprime preservando varianza global → t-SNE y UMAP preservan vecindarios a cambio de distorsionar lo global → ninguna figura se acepta sin una métrica y una prueba de estabilidad → ese espacio es la entrada del clustering de mañana.

**Tiempos sugeridos:** apertura + R (12 min) · PCA (25 min) · M + t-SNE (18 min) · UMAP (12 min) · evaluación (12 min) · hands-on y cierre (11 min).

---

## 4. Contenido slide por slide

### 4.A Apertura (A1–A3)

| Slide | Título | Contenido | Visual |
|---|---|---|---|
| A1 | Portada | "Reducción de dimensionalidad", Aprendizaje Automático · Sesión 5, autor y fecha 2026-2, branding EAFIT. | `\titleframe` |
| A2 | Contenido | `\tableofcontents`. | — |
| A3 | Dónde quedamos | S1–S4 tenían target. Hoy no hay $y$: la pregunta es qué estructura tiene $X$. Pipeline de la unidad: tabla → **representación (S5)** → grupos, anomalías y drift (S6). | Diagrama de 3 cajas |

### 4.1 Por qué reducir dimensión (R0–R5)

| Slide | Título | Contenido | Visual |
|---|---|---|---|
| R0 | Sección | "Por qué reducir dimensión". | `\sectionframe` |
| R1 | Una fila es un punto | Un vendedor de Olist = vector de 8 features. Distancia euclídea entre dos vendedores con números. Todo lo de hoy y mañana depende de esta distancia. | Tabla de 3 vendedores + $d(x,x')=\lVert x-x'\rVert_2$ |
| R2 | La maldición de la dimensionalidad | Concentración de distancias: $\frac{d_{\max}-d_{\min}}{d_{\min}} \to 0$ cuando $p$ crece. El vecino más cercano y el más lejano quedan casi a la misma distancia. | Curva del cociente vs $p$ (simulación uniforme) |
| R3 | Cuatro usos de la reducción | Visualizar · comprimir · quitar ruido y colinealidad · preprocesar para clustering o modelos. Cada uso pide un método distinto. | Tabla uso → método típico → qué se evalúa |
| R4 | Selección frente a extracción | **Seleccionar** columnas (interpretables, pierden combinaciones) frente a **construir** nuevas (PCA, embeddings: compactas, menos interpretables). | Dos columnas |
| R5 | El escalado decide la geometría | Sin escalar, "ventas en BRL" domina la distancia. Log para variables sesgadas, `StandardScaler` o `RobustScaler`. Regla: en no supervisado el preprocesamiento **es** parte del modelo. | Scatter antes/después |

### 4.2 PCA (P0–P10)

| Slide | Título | Contenido | Visual |
|---|---|---|---|
| P0 | Sección | "PCA: proyección lineal". | `\sectionframe` |
| P1 | La idea | Rotar los ejes hacia la dirección de máxima varianza y proyectar. PC2 es ortogonal a PC1, y así sucesivamente. | Nube 2D elongada con PC1/PC2 dibujados |
| P2 | El problema de optimización | $\mathbf{w}^*=\arg\max_{\lVert\mathbf{w}\rVert=1}\mathbf{w}^\top\Sigma\mathbf{w}$ → multiplicadores de Lagrange → $\Sigma\mathbf{w}=\lambda\mathbf{w}$. La varianza proyectada es $\lambda$. | Fórmulas anotadas |
| P3 | PCA a mano (mini-mundo) | Los 6 puntos → centrar → $\Sigma$ → $\lambda_1=14{,}16$, $\lambda_2=0{,}51$ → PC1 $\approx(0{,}79;0{,}61)$ → 96,5 % de varianza en una dimensión. Las coordenadas $z_1$ separan {A,B,C} de {D,E,F}. | Tabla paso a paso (sección 2.1) + recta PC1 en TikZ |
| P4 | PCA vía SVD | $X_c = U S V^\top$; componentes = columnas de $V$; $\lambda_j = s_j^2/(n-1)$. Por qué las librerías usan SVD: estabilidad numérica y SVD aleatorizada para $n$ grande. | Diagrama de matrices |
| P5 | Varianza explicada y scree plot | $\text{EVR}_j=\lambda_j/\sum_l\lambda_l$. Criterios: 80–90 % acumulado, codo del scree, o el $k$ que maximiza la tarea siguiente. | Scree + acumulada (Olist) |
| P6 | Leer los loadings | Qué variables forman PC1 y PC2 en Olist (p. ej. "tamaño del vendedor", "experiencia de entrega"). El signo es arbitrario; nombrar un componente es una hipótesis. | Heatmap de loadings |
| P7 | Reconstrucción | $\hat X = Z_k V_k^\top + \bar x$; el error de reconstrucción baja con $k$. Un punto mal reconstruido no se parece al resto: score de anomalía (puente a S6). | Fórmula + curva error vs $k$ |
| P8 | PCA dentro del pipeline | `fit` solo en train; PCA dentro de `Pipeline`; `n_components` como hiperparámetro en CV (conexión con S1–S2). PCA ajustado sobre todo el dataset = leakage. | Esquema del pipeline |
| P9 | Cuándo PCA falla | (1) estructura no lineal (swiss roll); (2) sensible a outliers; (3) varianza ≠ relevancia: la dirección de mayor varianza puede no servir para la tarea. | 3 mini-figuras |
| P10 | PCA en código | `StandardScaler` → `PCA(n_components=0.9)`; `explained_variance_ratio_`, `components_`, `inverse_transform`. | `lstlisting` corto |

### 4.3 Más allá de lo lineal (M0–M2)

| Slide | Título | Contenido | Visual |
|---|---|---|---|
| M0 | Sección | "Más allá de lo lineal". | `\sectionframe` |
| M1 | La hipótesis de la variedad | Datos de alta dimensión viven cerca de una superficie de baja dimensión. Swiss roll: dos puntos cercanos en euclídea pueden estar lejos sobre la superficie (geodésica). | Swiss roll 3D + versión desenrollada |
| M2 | Global frente a local | PCA preserva varianza global; t-SNE y UMAP preservan vecindarios. Mención: kernel PCA e Isomap como intermedios (no se profundiza). | Tabla método → qué preserva → qué distorsiona |

### 4.4 t-SNE (T0–T6)

| Slide | Título | Contenido | Visual |
|---|---|---|---|
| T0 | Sección | "t-SNE". | `\sectionframe` |
| T1 | Vecindades como probabilidades | $p_{j\mid i}\propto\exp\!\left(-\lVert x_i-x_j\rVert^2/2\sigma_i^2\right)$; simetrizada $p_{ij}=\frac{p_{j\mid i}+p_{i\mid j}}{2n}$. Un vecino cercano recibe probabilidad alta. | Fórmula + punto con vecinos sombreados |
| T2 | Perplexity | $\sigma_i$ se ajusta para que $\text{Perp}(P_i)=2^{H(P_i)}$ sea fija ≈ número efectivo de vecinos (típico 5–50). Zonas densas → $\sigma_i$ pequeño. | Fórmula + intuición |
| T3 | Student-t y el problema de aglomeración | En 2D no caben todos los vecinos. $q_{ij}\propto(1+\lVert y_i-y_j\rVert^2)^{-1}$: colas pesadas permiten alejar lo moderadamente lejano. | Densidad gaussiana vs t superpuestas |
| T4 | El objetivo: divergencia KL | $\text{KL}(P\Vert Q)=\sum_{i\neq j} p_{ij}\log\frac{p_{ij}}{q_{ij}}$. Asimétrica: castiga separar vecinos, casi no castiga juntar puntos lejanos. Descenso de gradiente, *early exaggeration*, inicialización PCA. | Fórmula anotada |
| T5 | Mismos datos, distinta perplexity | Los tamaños de los grupos y las distancias entre grupos cambian con el parámetro: no significan nada. | Grilla 1×4 (perplexity 5/30/50/100) |
| T6 | Lecturas permitidas y prohibidas | Sí: quién es vecino de quién. No: distancias entre grupos, tamaños, densidades, número de grupos. No tiene `transform` para datos nuevos; costo $O(n\log n)$ (Barnes–Hut). | Tabla de 2 columnas + `rulebox` |

### 4.5 UMAP (U0–U5)

| Slide | Título | Contenido | Visual |
|---|---|---|---|
| U0 | Sección | "UMAP". | `\sectionframe` |
| U1 | Un grafo de vecinos difuso | Grafo $k$-NN con pesos $w_{ij}=\exp\!\left(-(d_{ij}-\rho_i)/\sigma_i\right)$, con $\rho_i$ = distancia al vecino más cercano; simetrización por unión difusa. | Grafo $k$-NN pequeño |
| U2 | Optimizar el layout | Cross-entropy entre el grafo de alta dimensión y el de baja dimensión: atracción en las aristas, repulsión con *negative sampling*. | Fórmula + esquema de fuerzas |
| U3 | `n_neighbors` y `min_dist` | `n_neighbors`: local ↔ global. `min_dist`: qué tan apretados quedan los puntos. | Grilla 2×3 sobre Olist |
| U4 | UMAP frente a t-SNE | UMAP: más rápido, conserva algo más de estructura global, tiene `transform()` para datos nuevos, admite > 2 dimensiones y otras métricas. Ambos: estocásticos y no lineales. | Tabla comparativa |
| U5 | UMAP antes de clustering | Práctica común: UMAP a 5–10 dimensiones con `min_dist=0` → HDBSCAN (S6). Riesgo: crear grupos que no existen. Regla: validar los grupos en el espacio original. | `rulebox` |

### 4.6 Evaluar un embedding (E0–E5)

| Slide | Título | Contenido | Visual |
|---|---|---|---|
| E0 | Sección | "Evaluar un embedding". | `\sectionframe` |
| E1 | Lo visual no basta | Antes de mirar: qué variables, qué escalado, qué semilla, qué parámetros. Una figura sin esos cuatro datos no es reproducible. | Checklist + `rulebox` |
| E2 | Trustworthiness y continuity | Trustworthiness penaliza **falsos vecinos** (vecinos en el embedding que no lo eran); continuity penaliza **vecinos perdidos**. Rango 0–1. `sklearn.manifold.trustworthiness`. | Fórmula + diagrama |
| E3 | Preservación de $k$-vecinos | Fracción de los $k$-NN originales que siguen siendo $k$-NN en el embedding. Medida simple y fácil de explicar. | Tabla PCA / t-SNE / UMAP (Olist) |
| E4 | Estabilidad entre semillas | Correr 4–5 semillas y comparar vecindades. Un patrón que no se repite no existe. PCA es determinista; t-SNE y UMAP no. | Grilla de 4 semillas |
| E5 | Evaluación por tarea | Si hay una variable auxiliar (región, % de retraso), ¿un $k$-NN sobre el embedding la recupera? El embedding como feature de un modelo supervisado. | Tabla |

### 4.7 Hands-on Olist y cierre (Z1–Z6)

| Slide | Título | Contenido | Visual |
|---|---|---|---|
| Z1 | Hands-on: vendedores de Olist | (1) features por vendedor en SQL sobre DuckDB; (2) log + escalado; (3) PCA, t-SNE y UMAP; (4) colorear por región y por % de retraso; (5) trustworthiness y estabilidad. | `taskbox` |
| Z2 | Tres mapas, un dataset | PCA vs t-SNE vs UMAP lado a lado, misma coloración. | Figura 1×3 |
| Z3 | Comparación numérica | Trustworthiness, preservación de $k$-NN y tiempo de cómputo por método. | Tabla |
| Z4 | Práctica: ¿qué puede concluir? | UMAP muestra 4 islas; PCA muestra solapamiento; t-SNE muestra muchas islas pequeñas. ¿Qué se puede afirmar y qué no? (idea tomada del deck heredado, slide 34). | `taskbox` |
| Z5 | Lo que debe quedar claro | (1) el escalado es parte del modelo; (2) PCA = varianza global, lineal, interpretable; (3) t-SNE/UMAP = vecindarios, no distancias globales; (4) ningún embedding sin métrica y semillas; (5) ver grupos no es tener grupos. | Lista |
| Z6 | Puente a la S6 y referencias | Mañana se agrupa en este espacio. Refs: Jolliffe y Cadima (2016); van der Maaten y Hinton (2008); McInnes, Healy y Melville (2018); Wattenberg, Viégas y Johnson (2016), *How to Use t-SNE Effectively*; Venna y Kaski (2001). | Lista |

---

## 5. Trazabilidad: syllabus de la S5 → slides

| Concepto del syllabus | Slides |
|---|---|
| PCA | **P1–P10** |
| t-SNE | **T1–T6** |
| UMAP | **U1–U5** |
| Hands-on (1) PCA, t-SNE y UMAP sobre datos reales | **Z1–Z2** |
| Hands-on (2) evaluar embeddings visual y numéricamente | **E1–E5, Z3** |
| *(Necesario)* Maldición de la dimensionalidad y escalado | **R2, R5** |

| Requisito del Módulo C del taller | Slides |
|---|---|
| Transformaciones de variables sesgadas y escalamiento | R5 |
| Proyección 2D con PCA y UMAP o t-SNE | P10, U3, Z2 |

---

## 6. Convenciones para el LaTeX

- `\documentclass[aspectratio=169,12pt]{beamer}` + `\usepackage{eafitml}`; copiar el preámbulo de macros locales de `../04_ml_recommender_shap/04_ml_recommender_shap.tex` (`\tealframe`, `\bigmsg`, `goldbox`, `grayblock`, `mltable`, `\hdr`, `\lstset`).
- Cajas: `defbox`, `keybox`, `rulebox`, `taskbox`; marcos: `\titleframe`, `\sectionframe`, `\imageframe`.
- Portada: Andrés Vásquez Restrepo, `avasquezr3@eafit.edu.co`, SI7009 - 1 (5553), Universidad EAFIT, "2026-2 -- Medellín", logo `figures/img_p001_01.png` (copiar de `../04_ml_recommender_shap/figures/`).

---

## 7. Figuras a generar

| Figura | Fuente | Slide |
|---|---|---|
| Cociente de distancias vs $p$ | Simulación uniforme | R2 |
| Scatter sin escalar vs escalado | Olist | R5 |
| Nube 2D con PC1/PC2 | Sintética | P1 |
| Scree + varianza acumulada | Olist | P5 |
| Heatmap de loadings | Olist | P6 |
| Error de reconstrucción vs $k$ | Olist | P7 |
| Fallos de PCA (swiss roll, outlier, varianza ≠ relevancia) | Sintética | P9 |
| Swiss roll 3D + desenrollado | `make_swiss_roll` | M1 |
| Gaussiana vs t de Student | Analítica | T3 |
| Grilla de perplexity | Olist | T5 |
| Grilla `n_neighbors` × `min_dist` | Olist | U3 |
| Grilla de semillas | Olist | E4 |
| PCA / t-SNE / UMAP 1×3 | Olist | Z2 |

Mini-mundo (P3) y diagramas (A3, P4, T1, U1, U2): TikZ directo. Las figuras Olist salen de un script que lee `olist.duckdb` (construido con `homeworks/taller/build_olist_duckdb.py`) y escribe en `figures/`.

---

## 8. Pendientes para el docente

- [ ] Confirmar **vendedor** como unidad de análisis en clase (el taller usa cliente).
- [x] `umap-learn` ya está en `pyproject.toml`.
- [x] Script de figuras: `../make_figures_s5_s6.py`.
- [ ] Si se quiere, notebook `05_ml_reduccion_dimensionalidad.ipynb` con los mismos números.
- [x] Crear `05_ml_dimensionality_reduction.tex`.
- [ ] Copiar el PDF a `lectures/`.
- [ ] Decidir qué hacer con `../04_ml_unsupervisedlearning/` (archivar en `_historic/`).
- [ ] Añadir la fila de la sesión 5 en el `README.md` raíz.

## 9. Resumen de conteo

| Sección | Slides |
|---|---|
| Apertura | 3 |
| Por qué reducir dimensión | 6 |
| PCA | 11 |
| Más allá de lo lineal | 3 |
| t-SNE | 7 |
| UMAP | 6 |
| Evaluar un embedding | 6 |
| Hands-on y cierre | 6 |
| **Total** | **48** |
