# Sesión 06 — Plan de contenido slide por slide (clustering, anomalías y confiabilidad)

Plan para **crear desde cero** las slides de la sesión 6, la última del curso. Se basa en:

- **`syllabusML.pdf`** → Unidad 3. Esta sesión toma los temas de clustering de la sesión 5 del syllabus (K-Means, DBSCAN, HDBSCAN, silhouette, stability index) y los de la sesión 6 (Mini-Batch K-Means, jerárquico, Isolation Forest, LOF, calibración, drift). La reducción de dimensionalidad quedó en `../05_ml_dimensionality_reduction/README.md`.
- **Deck heredado** `../04_ml_unsupervisedlearning/04_ml_unsupervisedlearning.tex` → referencia de tono para "cluster no es segmento" y outliers. No cubre DBSCAN/HDBSCAN, jerárquico, Davies–Bouldin ni estabilidad.
- **Taller, Módulo C** (`homeworks/latex/ml_taller_20262.tex`, entrega 17/10/2026) → K-Means + otra familia, codo / silueta / Davies–Bouldin, ARI entre semillas, perfilamiento. Esta sesión es la última antes de la entrega.

> **Convención.** Cada slide tiene un código (A1, C1, K1, …). Todas las filas son `NUEVO`.
>
> **Tamaño:** **49 slides** para **~1 h 30 min**. Máximo permitido: 50.
>
> **Fecha de la clase:** sábado 10/10/2026, 8:00 a. m.–12:00 m. (1 h 30 de clase; el resto del bloque son exposiciones).

---

## 0. Estado actual

| Pieza | Estado |
|---|---|
| Slides de la S6 | ✅ `06_ml_clustering_anomalies.tex` → 50 páginas (49 del plan + «Gracias»). Compilar con `latexmk -pdf` (copia local de `eafitml.sty`). |
| Figuras | ✅ `../make_figures_s5_s6.py --db olist.duckdb` genera `figures/*.png` y `figures/results_s6.json`. Resultado en Olist: $k=3$ (silueta 0,20, ARI 0,87); drift real feb–mar 2018 (PSI 0,34). |
| Notebook | ⚪ Fuera de alcance por ahora. `notebooks/ml_unsupervisedlearning.ipynb` (heredado) tiene IF/LOF y drift reutilizables. |
| Branding | ✅ `../eafitml.sty`. |

---

## 1. Qué exige el syllabus

Objetivos de la unidad que cubre esta sesión:

- implementar **K-Means** y **Mini-Batch K-Means** para clustering escalable;
- aplicar **DBSCAN, HDBSCAN** y **clustering jerárquico**;
- hacer **segmentación de clientes** con DR + clustering;
- detectar **outliers** y problemas de **calibración**;
- detectar **drift** en datos y en el comportamiento del modelo.

| Bloque | Concepto del syllabus |
|---|---|
| Centroides | K-Means baseline · K-Means vs Mini-Batch (escalabilidad, actualizaciones estocásticas) |
| Jerárquico | Hierarchical clustering |
| Densidad | DBSCAN & HDBSCAN |
| Validación | Silhouette score & stability index |
| Anomalías | Isolation Forest, LOF |
| Confiabilidad | Calibration curves, isotonic regression · drift detection (feature drift, prediction drift) |
| Hands-on | Segmentación DR → K-Means / Mini-Batch → interpretación · grupos de alto valor, riesgo o anómalos · comparar algoritmos · IF y LOF · diagnósticos de confiabilidad |

### 1.1 Decisiones de alcance

| Tema | Tratamiento | Motivo |
|---|---|---|
| Calibración | **1 slide de repaso** (R3) | Se vio a fondo en la S1 (curvas de confiabilidad, Platt, isotónica, Brier). |
| Drift | **Bloque corto** (R1–R2) | Tipos de drift + PSI/KS. Suficiente para la sección de limitaciones del taller. |
| GMM | Solo mención en C3 | El taller lo acepta como "otra familia", pero no cabe en 1 h 30. |

### 1.2 Conexión con el resto del curso

| Sesión | Concepto que reaparece |
|---|---|
| S1 | Calibración y Brier (R3) · no tunear con el test → no elegir $k$ mirando solo una métrica |
| S2 | Isolation Forest es un ensemble de árboles aleatorios |
| S3 | Ventanas de referencia vs actual para drift (orden temporal) |
| S5 | Escalado, PCA y UMAP como entrada del clustering · error de reconstrucción como anomalía |

---

## 2. Decisiones de diseño

| Decisión | Propuesta | Motivo |
|---|---|---|
| **Dataset** | **Olist, vendedores** (las mismas features de la S5). | Continuidad con la S5 y con la base del taller; el taller pide clientes, así que el método se enseña sin resolver el Módulo C. |
| **Mini-mundo** | Los 6 puntos de la S5 — A(1,2), B(2,1), C(2,3), D(6,5), E(7,7), F(8,6) — más **G(5,1)** para DBSCAN y outliers. | Lloyd, silueta y DBSCAN se calculan a mano con números verificables. |
| **Sintéticos** | `make_blobs`, `make_moons`, blobs anisotrópicos y de densidad variable. | La grilla "algoritmo × forma" muestra los supuestos de cada familia mejor que cualquier dato real. |
| **Librerías** | `scikit-learn` ≥ 1.3 (`KMeans`, `MiniBatchKMeans`, `AgglomerativeClustering`, `DBSCAN`, `HDBSCAN`, métricas, `IsolationForest`, `LocalOutlierFactor`), `scipy.cluster.hierarchy` (dendrograma), `scipy.stats.ks_2samp`. | Sin dependencias nuevas. |

### 2.1 Números del mini-mundo (verificados con NumPy)

| Cálculo | Valor |
|---|---|
| Lloyd, $k=2$, init "mala" en A y B | Iter. 1: {A,C,E} / {B,D,F} → centroides (3,33; 4,00) y (5,33; 4,00). Iter. 2: {A,B,C} / {D,E,F} → centroides (1,67; 2,00) y (7,00; 6,00). Iter. 3: sin cambios → converge. |
| Inercia final | $J = 6{,}67$ |
| Silueta de A | $a(A)=1{,}41$ (media a B y C) · $b(A)=7{,}23$ (media a D, E, F) · $s(A)=\frac{7{,}23-1{,}41}{7{,}23}=0{,}80$ |
| DBSCAN `eps`=2,3, `min_samples`=3 | A–F son núcleo (cada uno tiene 2 vecinos a ≤ 2,24); G tiene su vecino más cercano a 3,0 → **ruido**. Dos clusters: {A,B,C}, {D,E,F}. |

---

## 3. Estructura de la sesión 6

```
A. Apertura                                    A1–A3    3
1. Formular el clustering                      C0–C3    4
2. K-Means y Mini-Batch K-Means                K0–K7    8
3. Clustering jerárquico                       H0–H4    5
4. Clustering por densidad: DBSCAN y HDBSCAN   D0–D5    6
5. Validar sin etiquetas                       V0–V6    7
6. Segmentación accionable (Olist)             S0–S5    6
7. Outliers: Isolation Forest y LOF            O0–O3    4
8. Confiabilidad: drift y calibración          R0–R3    4
9. Cierre del curso                            Z1–Z2    2
                                               Total   49
```

**Hilo narrativo:** un cluster no es un segmento → la distancia y la familia de algoritmo definen qué grupos se pueden encontrar → sin $y$, se valida con métricas internas **y** estabilidad → un segmento es un grupo nombrable, estable y accionable → lo que no encaja en ningún grupo (outliers) → lo que cambia después de desplegar (drift) → cierre del curso.

**Tiempos sugeridos:** apertura + C (8 min) · K-Means (16 min) · jerárquico (9 min) · densidad (12 min) · validación (15 min) · segmentación (12 min) · outliers (8 min) · drift y calibración (7 min) · cierre (3 min).

---

## 4. Contenido slide por slide

### 4.A Apertura (A1–A3)

| Slide | Título | Contenido | Visual |
|---|---|---|---|
| A1 | Portada | "Clustering, anomalías y confiabilidad", Aprendizaje Automático · Sesión 6, autor y fecha 2026-2, branding EAFIT. | `\titleframe` |
| A2 | Contenido | `\tableofcontents`. | — |
| A3 | Dónde quedamos | Ayer: un espacio donde la distancia significa algo. Hoy: proponer grupos en ese espacio y decidir si sirven. Conexión explícita con el Módulo C del taller (entrega 17/10). | Dos columnas |

### 4.1 Formular el clustering (C0–C3)

| Slide | Título | Contenido | Visual |
|---|---|---|---|
| C0 | Sección | "Formular el clustering". | `\sectionframe` |
| C1 | Un cluster no es un segmento | El algoritmo entrega una partición. El negocio necesita grupos **nombrables, estables, accionables y de tamaño útil**. Siempre hay clusters; no siempre hay segmentos. | `defbox` vs `rulebox` |
| C2 | La distancia define los grupos | Escalado, log y variables dominantes (repaso de R5 de la S5). Caso Olist: la mayoría de los clientes compra una sola vez → la frecuencia es casi constante y no separa a nadie. | Ejemplo numérico |
| C3 | Cuatro familias | Centroides (K-Means), jerárquico (Ward, linkage), densidad (DBSCAN, HDBSCAN), modelos (GMM, solo mención). Cada familia supone una forma de cluster. | Tabla familia → supuesto → dónde falla |

### 4.2 K-Means y Mini-Batch K-Means (K0–K7)

| Slide | Título | Contenido | Visual |
|---|---|---|---|
| K0 | Sección | "K-Means". | `\sectionframe` |
| K1 | El objetivo | Inercia $J=\sum_{k=1}^{K}\sum_{i\in C_k}\lVert x_i-\mu_k\rVert^2$. Minimizarla exactamente es NP-hard; el algoritmo de Lloyd encuentra un mínimo local. | Fórmula anotada |
| K2 | Lloyd a mano (mini-mundo) | Init en A y B → asignar → actualizar → converge en 3 iteraciones a {A,B,C} / {D,E,F}, $J=6{,}67$ (sección 2.1). | Tabla + 3 paneles TikZ |
| K3 | Inicialización: k-means++ | Siguiente centroide con probabilidad $\propto D(x)^2$. `n_init` repite y se queda con la menor inercia. Semillas distintas pueden dar particiones distintas (puente a V5). | Init mala vs k-means++ |
| K4 | Supuestos y fallos | Clusters convexos y esféricos, varianza parecida, tamaños parecidos. Falla con moons, grupos anisotrópicos y outliers (que arrastran centroides). | Grilla 1×3 de fallos |
| K5 | Elegir $k$: el codo | Inercia vs $k$. Siempre baja; el codo rara vez es nítido. Se combina con silueta, Davies–Bouldin, estabilidad y criterio de negocio (V6). | Curva de codo (Olist) |
| K6 | Mini-Batch K-Means | Cada iteración usa un lote de $b$ puntos: $\mu_k \leftarrow (1-\eta)\mu_k+\eta\,x$, con $\eta=1/n_k$ (conteo acumulado del centroide). Costo por iteración $O(bKp)$ en lugar de $O(nKp)$. | Pseudocódigo |
| K7 | K-Means frente a Mini-Batch | Tiempo vs inercia relativa al crecer $n$ (sintético hasta $10^6$). Vale la pena con $n$ grande o datos en flujo (`partial_fit`); la pérdida de inercia suele ser de pocos puntos porcentuales. | Gráfico tiempo/inercia |

### 4.3 Clustering jerárquico (H0–H4)

| Slide | Título | Contenido | Visual |
|---|---|---|---|
| H0 | Sección | "Clustering jerárquico". | `\sectionframe` |
| H1 | Aglomerativo | Cada punto empieza como su propio cluster; se fusiona el par más cercano hasta que queda uno. Mini-mundo (single linkage): A–B, A–C y E–F se fusionan a 1,41; D se une a {E,F} a 2,24; los dos grupos al final. | 4 pasos en TikZ |
| H2 | Linkage: cómo se mide la distancia entre grupos | Single (mínima: forma cadenas), complete (máxima), average, **Ward** (menor aumento de inercia: lo más parecido a K-Means). | Tabla + fórmulas |
| H3 | Leer y cortar el dendrograma | Altura = disimilitud de la fusión. Se corta por altura o por $k$; un salto grande en altura sugiere $k$. | Dendrograma (muestra de Olist) |
| H4 | Costo y uso práctico | Memoria $O(n^2)$: con ~3 k vendedores va bien; con 100 k clientes no → muestra o `connectivity`. Útil para ver la jerarquía de segmentos (segmentos dentro de segmentos). | `goldbox` |

### 4.4 Clustering por densidad: DBSCAN y HDBSCAN (D0–D5)

| Slide | Título | Contenido | Visual |
|---|---|---|---|
| D0 | Sección | "Clustering por densidad". | `\sectionframe` |
| D1 | DBSCAN | `eps` y `min_samples`; puntos **núcleo**, **borde** y **ruido**; un cluster = puntos núcleo conectados más sus bordes. No pide $k$. Mini-mundo: dos clusters y G es ruido (sección 2.1). | Diagrama TikZ de los 3 tipos |
| D2 | Elegir `eps` | Gráfico de la distancia al $k$-ésimo vecino, ordenada; `eps` cerca del codo. Limitación: un solo `eps` supone una sola densidad. | Curva de $k$-distancia (Olist) |
| D3 | HDBSCAN | Recorre todos los `eps` a la vez: distancia de alcanzabilidad mutua → árbol de expansión mínima → jerarquía → árbol condensado → se eligen los clusters más persistentes. Parámetro principal: `min_cluster_size`. Entrega `probabilities_`. `sklearn.cluster.HDBSCAN` (≥ 1.3). | DBSCAN (2 `eps`) vs HDBSCAN |
| D4 | El ruido es un resultado | La etiqueta `-1` no es un error: son candidatos a outlier o casos "no segmentables". Hay que reportar qué porcentaje es ruido y decidir qué se hace con él. | `keybox` |
| D5 | Cuatro algoritmos, mismos datos | Blobs / moons / anisotrópico / densidad variable × K-Means / Ward / DBSCAN / HDBSCAN. | Grilla 4×4 |

### 4.5 Validar sin etiquetas (V0–V6)

| Slide | Título | Contenido | Visual |
|---|---|---|---|
| V0 | Sección | "Validar sin etiquetas". | `\sectionframe` |
| V1 | Silueta | $s(i)=\frac{b(i)-a(i)}{\max\{a(i),b(i)\}}$; $a$ = distancia media a su cluster, $b$ = al cluster vecino más cercano. Mini-mundo: $s(A)=0{,}80$. Rango −1 a 1. | Fórmula + cálculo |
| V2 | El gráfico de silueta | Por cluster: el grosor es el tamaño; los valores negativos son puntos mal asignados. Más informativo que el promedio. | Silhouette plot (Olist) |
| V3 | Davies–Bouldin y Calinski–Harabasz | DB: promedio del peor cociente dispersión/separación (menor es mejor). CH: varianza entre / dentro (mayor es mejor). | Tabla métrica → dirección → sesgo |
| V4 | Las métricas internas prefieren esferas | En moons, la silueta premia a K-Means sobre DBSCAN aunque DBSCAN acierta. Con ruido: excluir `-1` y reportar el % de ruido. | Ejemplo moons con valores |
| V5 | Estabilidad: ARI | Reajustar con otras semillas o submuestras bootstrap y comparar particiones con el *Adjusted Rand Index* (0 = azar, 1 = idénticas). *Stability index* = ARI medio entre réplicas (Ben-Hur et al., 2002; Lange et al., 2004). | Fórmula + boxplot de ARI por $k$ |
| V6 | Decidir $k$ con evidencia | Tabla $k$ = 3…7: inercia, silueta, DB, ARI medio y tamaño del cluster más pequeño → elección justificada, más el criterio de negocio. | Tabla de decisión (Olist) |

### 4.6 Segmentación accionable — Olist (S0–S5)

| Slide | Título | Contenido | Visual |
|---|---|---|---|
| S0 | Sección | "De cluster a segmento". | `\sectionframe` |
| S1 | El pipeline completo | SQL → features por vendedor → log y escalado → (PCA) → clustering → validación → perfilamiento → acción. Qué se ajusta con qué datos; las proyecciones 2D solo se usan para mirar. | Diagrama horizontal |
| S2 | Perfilamiento | Mediana por cluster dividida por la mediana global (índice > 1 = por encima de lo normal), tamaño de cada cluster y % de ruido. | Heatmap de índices |
| S3 | Nombrar y actuar | Ejemplos: "grandes y puntuales", "pequeños con retrasos", "nicho bien calificado" → evidencia → acción (soporte logístico, programa de fidelización, auditoría). | Tabla segmento → evidencia → acción |
| S4 | Práctica: ¿aprobaría esta segmentación? | Silueta 0,55 pero ARI medio 0,40 y un cluster con el 85 % de los vendedores. ¿Se aprueba? ¿Qué pediría antes? | `taskbox` |
| S5 | Anti-patrones | Elegir $k$ solo por silueta · agrupar sobre t-SNE en 2D · no escalar · ignorar la estabilidad · segmentos sin acción distinta · un cluster gigante y varios diminutos. | Lista + `rulebox` |

### 4.7 Outliers: Isolation Forest y LOF (O0–O3)

| Slide | Título | Contenido | Visual |
|---|---|---|---|
| O0 | Sección | "Lo que no encaja: outliers". | `\sectionframe` |
| O1 | Isolation Forest | Árboles con cortes aleatorios: un punto raro se aísla en pocos cortes. Score $s(x)=2^{-E[h(x)]/c(n)}$; cerca de 1 = anómalo. `contamination` fija el umbral. Conexión con los ensembles de la S2. | Cortes aleatorios aislando a G |
| O2 | Local Outlier Factor | Compara la densidad local de un punto con la de sus vecinos: LOF ≈ 1 normal, LOF ≫ 1 anómalo local. IF detecta lo raro en general; LOF, lo raro para su vecindario. | Diagrama de densidades + tabla IF vs LOF |
| O3 | Práctica: el 1 % más raro | Vendedores marcados por IF, LOF o ruido de HDBSCAN: ¿error de dato, fraude, vendedor estrella? Se investiga antes de borrar. | `taskbox` |

### 4.8 Confiabilidad: drift y calibración (R0–R3)

| Slide | Título | Contenido | Visual |
|---|---|---|---|
| R0 | Sección | "Cuando el mundo cambia". | `\sectionframe` |
| R1 | Tipos de drift | Covariate $P(X)$, prior $P(y)$, concepto $P(y\mid X)$ y drift de predicciones. En segmentación: vendedores que migran de cluster y clusters que cambian de tamaño. | Tabla tipo → qué cambia → cómo se detecta |
| R2 | Detectarlo: PSI y KS | $\text{PSI}=\sum_b (a_b-e_b)\ln(a_b/e_b)$ con umbrales habituales < 0,1 / 0,1–0,25 / > 0,25; prueba KS para variables continuas. Ventana de referencia vs actual (Olist 2017 vs primer semestre de 2018). | Fórmula + histogramas superpuestos |
| R3 | Repaso: calibración y plan de monitoreo | Curva de confiabilidad y Brier (S1) como chequeo periódico. Tabla: qué monitorear (features, predicciones, calibración, tamaños de segmento), con qué frecuencia y qué acción dispara. | Tabla de monitoreo |

### 4.9 Cierre del curso (Z1–Z2)

| Slide | Título | Contenido | Visual |
|---|---|---|---|
| Z1 | El curso en una página | S1 evaluar bien → S2 boosting y HPO → S3 el tiempo → S4 ranking y SHAP → S5 representar → S6 estructura y confiabilidad. Una regla por sesión. | Diagrama de 6 pasos |
| Z2 | Taller, examen y referencias | Checklist del Módulo C (clientes, ≥ 2 familias, codo/silueta/DB, PCA + UMAP, ARI, perfiles y acción). Examen final. Refs: Ester et al. (1996); Campello, Moulavi y Sander (2013); Liu, Ting y Zhou (2008); Breunig et al. (2000); Hubert y Arabie (1985); guía de clustering de scikit-learn. | Lista |

---

## 5. Trazabilidad: syllabus → slides

| Concepto del syllabus | Slides |
|---|---|
| K-Means baseline | **K1–K5** |
| K-Means vs Mini-Batch K-Means | **K6–K7** |
| Hierarchical clustering | **H1–H4** |
| DBSCAN & HDBSCAN | **D1–D5** |
| Silhouette score & stability index | **V1–V2, V5** |
| Isolation Forest, LOF | **O1–O3** |
| Calibration curves, isotonic regression | **R3** (repaso de la S1) |
| Drift detection (feature, prediction) | **R1–R2** |
| Hands-on: segmentación DR → clustering → interpretación | **S1–S3** |
| Hands-on: grupos de alto valor, riesgo o anómalos | **S3, O3** |
| Hands-on: comparar algoritmos | **D5, V4** |

| Requisito del Módulo C del taller | Slides |
|---|---|
| K-Means + otra familia | K*, H*, D* |
| Codo, silueta y Davies–Bouldin | K5, V1–V3 |
| Criterio de negocio para $k$ | V6, S3 |
| Estabilidad con ARI | V5 |
| Perfilamiento y acción por segmento | S2–S3 |

---

## 6. Convenciones para el LaTeX

Las mismas de `../05_ml_dimensionality_reduction/README.md` (sección 6): `eafitml.sty`, preámbulo de macros de S4, portada de S4 con "Sesión 6".

---

## 7. Figuras a generar

| Figura | Fuente | Slide |
|---|---|---|
| Init mala vs k-means++ | Sintética | K3 |
| Fallos de K-Means (moons, anisotrópico, outlier) | Sintética | K4 |
| Curva de codo | Olist | K5 |
| Tiempo/inercia K-Means vs Mini-Batch | Sintética, $n$ hasta $10^6$ | K7 |
| Dendrograma | Muestra Olist | H3 |
| Curva de $k$-distancia | Olist | D2 |
| DBSCAN (2 `eps`) vs HDBSCAN con densidades distintas | Sintética | D3 |
| Grilla 4×4 algoritmo × forma | Sintética | D5 |
| Gráfico de silueta | Olist | V2 |
| Silueta en moons | Sintética | V4 |
| Boxplot de ARI por $k$ | Olist | V5 |
| Heatmap de perfiles | Olist | S2 |
| Histogramas de referencia vs actual | Olist 2017 vs 2018 | R2 |

Mini-mundo (K2, H1, D1, O1) y diagramas (S1, Z1): TikZ directo.

---

## 8. Pendientes para el docente

- [ ] Confirmar **vendedor** como unidad de análisis en clase.
- [x] `scikit-learn>=1.5` en `pyproject.toml` (incluye `HDBSCAN`).
- [x] Script de figuras: `../make_figures_s5_s6.py`.
- [x] Crear `06_ml_clustering_anomalies.tex`.
- [ ] Copiar el PDF a `lectures/`.
- [ ] Añadir la fila de la sesión 6 en el `README.md` raíz.

## 9. Resumen de conteo

| Sección | Slides |
|---|---|
| Apertura | 3 |
| Formular el clustering | 4 |
| K-Means y Mini-Batch | 8 |
| Jerárquico | 5 |
| Densidad | 6 |
| Validar sin etiquetas | 7 |
| Segmentación accionable | 6 |
| Outliers | 4 |
| Drift y calibración | 4 |
| Cierre | 2 |
| **Total** | **49** |
