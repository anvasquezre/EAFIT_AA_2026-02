"""Figuras y números de las sesiones 5 (reducción de dimensionalidad) y 6 (clustering).

Lee la base Olist en DuckDB (construida con homeworks/taller/build_olist_duckdb.py),
arma una tabla de vendedores con SQL y genera todas las figuras de los decks en
05_ml_dimensionality_reduction/figures y 06_ml_clustering_anomalies/figures.
Los números citados en las slides quedan en figures/results_s5.json y results_s6.json.

Uso:
    uv run python lectures/latex/make_figures_s5_s6.py --db data/olist.duckdb
    uv run python lectures/latex/make_figures_s5_s6.py --db data/olist.duckdb --only s6
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import ks_2samp
from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs, make_moons, make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
OUT5 = HERE / "05_ml_dimensionality_reduction" / "figures"
OUT6 = HERE / "06_ml_clustering_anomalies" / "figures"
SEED = 42

# ------------------------------------------------------------------ estilo
INK, INK2, GRID = "#333333", "#656565", "#DBDBDC"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#8c5ac8"]
NOISE = "#b5b5b5"
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f2f2f2", "#eb6834"])

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "lines.linewidth": 2,
})


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", facecolor="white")
    plt.close(fig)
    print(f"  {out.parent.name}/figures/{name}.png")


def clean_ax(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def colors_for(labels):
    return [NOISE if l < 0 else SERIES[l % len(SERIES)] for l in labels]


# ------------------------------------------------------------------ datos Olist
SELLERS_SQL = """
WITH items AS (
    SELECT oi.seller_id, oi.order_id, oi.price, oi.freight_value, p.product_category_name AS cat
    FROM order_items oi
    JOIN orders o USING (order_id)
    LEFT JOIN products p USING (product_id)
    WHERE o.order_status = 'delivered'
      AND o.order_purchase_timestamp < TIMESTAMP '2018-06-01'
),
ord AS (
    SELECT DISTINCT i.seller_id, o.order_id,
           o.order_delivered_customer_date > o.order_estimated_delivery_date AS late,
           date_diff('hour', o.order_approved_at, o.order_delivered_carrier_date) / 24.0 AS handling_days
    FROM items i JOIN orders o USING (order_id)
),
rev AS (
    SELECT order_id, avg(review_score) AS score FROM order_reviews GROUP BY order_id
)
SELECT s.seller_id,
       any_value(se.seller_state)                    AS state,
       sum(s.price)                                  AS revenue,
       count(DISTINCT s.order_id)                    AS n_orders,
       sum(s.price) / count(DISTINCT s.order_id)     AS ticket,
       sum(s.freight_value) / sum(s.price)           AS freight_ratio,
       count(DISTINCT s.cat)                         AS n_categories,
       (SELECT avg(late::INT) FROM ord WHERE ord.seller_id = s.seller_id)          AS late_rate,
       (SELECT median(handling_days) FROM ord WHERE ord.seller_id = s.seller_id)   AS handling_days,
       (SELECT avg(r.score) FROM ord JOIN rev r USING (order_id)
         WHERE ord.seller_id = s.seller_id)                                        AS review_mean
FROM items s JOIN sellers se USING (seller_id)
GROUP BY s.seller_id
HAVING count(DISTINCT s.order_id) >= 3
"""

FEATURES = ["revenue", "n_orders", "ticket", "freight_ratio", "n_categories",
            "late_rate", "handling_days", "review_mean"]
LOG_FEATURES = ["revenue", "n_orders", "ticket", "n_categories", "handling_days"]
LABELS_ES = {
    "revenue": "ventas", "n_orders": "nº órdenes", "ticket": "ticket medio",
    "freight_ratio": "flete / precio", "n_categories": "nº categorías",
    "late_rate": "% retraso", "handling_days": "días a transportadora", "review_mean": "calificación",
}
REGION = {
    "SP": "Sudeste", "RJ": "Sudeste", "MG": "Sudeste", "ES": "Sudeste",
    "PR": "Sur", "SC": "Sur", "RS": "Sur",
    "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MT": "Centro-Oeste", "MS": "Centro-Oeste",
}


def load_sellers(db: Path) -> pd.DataFrame:
    con = duckdb.connect(str(db), read_only=True)
    df = con.sql(SELLERS_SQL).df()
    con.close()
    # GROUP BY no garantiza orden: ordenar para que submuestras y semillas sean reproducibles
    df = df.dropna(subset=FEATURES).sort_values("seller_id").reset_index(drop=True)
    df["handling_days"] = df["handling_days"].clip(lower=0)
    df["region"] = df["state"].map(REGION).fillna("Norte/Nordeste")
    return df


def prepare(df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURES].copy()
    for c in LOG_FEATURES:
        X[c] = np.log1p(X[c])
    return StandardScaler().fit_transform(X)


def knn_preservation(X, Y, k=10):
    a = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(return_distance=False)[:, 1:]
    b = NearestNeighbors(n_neighbors=k + 1).fit(Y).kneighbors(return_distance=False)[:, 1:]
    return float(np.mean([len(set(a[i]) & set(b[i])) / k for i in range(len(X))]))


def run_umap(X, seed=SEED, **kw):
    import umap  # import tardío: tarda en cargar

    return umap.UMAP(random_state=seed, **kw).fit_transform(X)


def scatter_by_size(ax, Y, df, title):
    ax.scatter(Y[:, 0], Y[:, 1], s=7, c=np.log10(df["revenue"]), cmap=BLUES, linewidths=0)
    ax.set_title(title)
    clean_ax(ax)


def scatter_by_region(ax, Y, df, title, legend=False):
    regions = ["Sudeste", "Sur", "Centro-Oeste", "Norte/Nordeste"]
    for i, r in enumerate(regions):
        m = (df["region"] == r).to_numpy()
        ax.scatter(Y[m, 0], Y[m, 1], s=7, c=SERIES[i], alpha=0.75, linewidths=0, label=r)
    ax.set_title(title)
    clean_ax(ax)
    if legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, markerscale=3, fontsize=11)


# ------------------------------------------------------------------ S5
def s5(df: pd.DataFrame, X: np.ndarray) -> dict:
    out, res = OUT5, {}
    rng = np.random.default_rng(SEED)
    res["n_sellers"] = len(df)
    res["three_sellers"] = df.sort_values("revenue").iloc[[len(df) // 10, len(df) // 2, len(df) - 5]][
        ["seller_id"] + FEATURES].round(3).to_dict("records")

    # R2: concentración de distancias
    ps = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    ratio = []
    for p in ps:
        Z = rng.uniform(size=(500, p))
        d = np.linalg.norm(Z[1:] - Z[0], axis=1)
        ratio.append((d.max() - d.min()) / d.min())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ps, ratio, color=SERIES[0], marker="o", ms=7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("dimensión $p$"); ax.set_ylabel(r"$(d_{\max}-d_{\min})/d_{\min}$")
    ax.set_title("500 puntos uniformes: el más lejano y el más cercano se parecen")
    save(fig, out, "r2_distance_concentration")
    res["distance_ratio"] = dict(zip(map(str, ps), np.round(ratio, 3).tolist()))

    # R5: escalado
    raw = df[["revenue", "late_rate"]].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(raw[:, 0], raw[:, 1], s=7, c=SERIES[0], alpha=.6, linewidths=0)
    axes[0].set_xlabel("ventas (BRL)"); axes[0].set_ylabel("% retraso"); axes[0].set_title("Sin transformar")
    sc = np.c_[np.log1p(raw[:, 0]), raw[:, 1]]
    sc = StandardScaler().fit_transform(sc)
    axes[1].scatter(sc[:, 0], sc[:, 1], s=7, c=SERIES[0], alpha=.6, linewidths=0)
    axes[1].set_xlabel("log(ventas), estandarizado"); axes[1].set_ylabel("% retraso, estandarizado")
    axes[1].set_title("log + StandardScaler")
    axes[1].set_aspect("equal", adjustable="datalim")
    save(fig, out, "r5_scaling")

    # P1: idea de PCA
    Z = rng.multivariate_normal([0, 0], [[3, 2.2], [2.2, 2]], size=300)
    pca2 = PCA().fit(Z)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(Z[:, 0], Z[:, 1], s=10, c=SERIES[0], alpha=.45, linewidths=0)
    for j, (vec, ev) in enumerate(zip(pca2.components_, pca2.explained_variance_)):
        v = vec * 2 * np.sqrt(ev)
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=2.5, color=[SERIES[1], SERIES[2]][j]))
        ax.text(*(v * 1.15), f"PC{j+1}", color=INK, fontsize=13, weight="bold", ha="center")
    ax.set_aspect("equal"); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    save(fig, out, "p1_pca_idea")

    # P5: scree
    pca = PCA().fit(X)
    evr = pca.explained_variance_ratio_
    res["evr"] = np.round(evr, 4).tolist()
    res["evr_cum"] = np.round(np.cumsum(evr), 4).tolist()
    k = np.arange(1, len(evr) + 1)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(k, evr, color=SERIES[0], width=0.6, label="por componente")
    ax.plot(k, np.cumsum(evr), color=SERIES[1], marker="o", ms=7, label="acumulada")
    ax.axhline(0.8, color=INK2, ls="--", lw=1)
    ax.text(len(evr) + 0.4, 0.8, "80 %", va="center", color=INK2)
    ax.set_xticks(k); ax.set_xlabel("componente"); ax.set_ylabel("varianza explicada")
    ax.set_ylim(0, 1.05); ax.legend(loc="center right")
    ax.set_title(f"PCA sobre {len(df):,} vendedores de Olist (8 variables)".replace(",", "."))
    save(fig, out, "p5_scree")

    # P6: loadings
    L = pca.components_[:3].T
    for j in range(L.shape[1]):  # signo: que la variable de mayor |carga| sea positiva
        L[:, j] *= np.sign(L[np.argmax(np.abs(L[:, j])), j])
    res["loadings"] = {FEATURES[i]: np.round(L[i], 3).tolist() for i in range(len(FEATURES))}
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    im = ax.imshow(L, cmap=DIVERGING, vmin=-0.75, vmax=0.75, aspect="auto")
    ax.set_xticks(range(3), [f"PC{j+1}\n({evr[j]:.0%})" for j in range(3)])
    ax.set_yticks(range(len(FEATURES)), [LABELS_ES[f] for f in FEATURES])
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            ax.text(j, i, f"{L[i, j]:.2f}", ha="center", va="center", fontsize=13, color=INK)
    ax.grid(False)
    save(fig, out, "p6_loadings")

    # P7: reconstrucción
    errs = []
    for kk in range(1, len(FEATURES) + 1):
        p = PCA(n_components=kk).fit(X)
        errs.append(float(np.mean((X - p.inverse_transform(p.transform(X))) ** 2)))
    res["recon_mse"] = np.round(errs, 4).tolist()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(errs) + 1), errs, color=SERIES[0], marker="o", ms=7)
    ax.set_xlabel("componentes conservados $k$"); ax.set_ylabel("MSE de reconstrucción")
    ax.set_xticks(range(1, len(errs) + 1))
    save(fig, out, "p7_reconstruction")

    # P9: fallos de PCA
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    t = rng.uniform(0, 2 * np.pi, 300)
    circ = np.c_[np.cos(t), np.sin(t)] * np.where(rng.uniform(size=300) < .5, 1, 3)[:, None] + rng.normal(0, .1, (300, 2))
    axes[0].scatter(circ[:, 0], circ[:, 1], s=8, c=SERIES[0], alpha=.6, linewidths=0)
    axes[0].set_title("No lineal: anillos"); axes[0].set_aspect("equal")
    Z = rng.multivariate_normal([0, 0], [[1, .0], [0, .15]], 200)
    Zo = np.r_[Z, [[0, 9], [0.3, 8.5], [-0.2, 9.3]]]
    for data, col, lab in [(Z, SERIES[1], "sin outliers"), (Zo, SERIES[0], "con 3 outliers")]:
        v = PCA(1).fit(data).components_[0] * 4
        axes[1].plot([-v[0], v[0]], [-v[1], v[1]], color=col, lw=2.5, label=lab)
    axes[1].scatter(Zo[:, 0], Zo[:, 1], s=8, c=INK2, alpha=.6, linewidths=0)
    axes[1].set_title("Outliers giran PC1"); axes[1].legend(fontsize=10, loc="lower right"); axes[1].set_aspect("equal")
    a = rng.normal(0, 3, 300); b = rng.normal(0, .4, 300); y = (b > 0).astype(int)
    axes[2].scatter(a[y == 0], b[y == 0], s=8, c=SERIES[0], linewidths=0, label="clase 0")
    axes[2].scatter(a[y == 1], b[y == 1], s=8, c=SERIES[1], linewidths=0, label="clase 1")
    axes[2].set_ylim(-4, 4); axes[2].set_title("PC1 (horizontal) ignora la clase")
    axes[2].legend(fontsize=10, loc="upper right")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    save(fig, out, "p9_pca_failures")

    # M1: swiss roll
    S, tt = make_swiss_roll(3000, noise=0.1, random_state=SEED)
    fig = plt.figure(figsize=(12, 4.5))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.scatter(S[:, 0], S[:, 1], S[:, 2], c=tt, cmap=BLUES, s=5)
    ax.set_title("Swiss roll en 3D"); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(10, -70)
    P = PCA(2).fit_transform(S)
    ax = fig.add_subplot(1, 3, 2); ax.scatter(P[:, 0], P[:, 1], c=tt, cmap=BLUES, s=5); ax.set_title("PCA 2D: sigue enrollado"); clean_ax(ax)
    U = run_umap(S, n_neighbors=30)
    ax = fig.add_subplot(1, 3, 3); ax.scatter(U[:, 0], U[:, 1], c=tt, cmap=BLUES, s=5); ax.set_title("UMAP 2D: desenrolla (por tramos)"); clean_ax(ax)
    save(fig, out, "m1_swissroll")

    # T3: gaussiana vs t
    d = np.linspace(0, 5, 300)
    g = np.exp(-d ** 2 / 2); st = 1 / (1 + d ** 2)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(d, g / g[0], color=SERIES[0], label="gaussiana (alta dim.)")
    ax.plot(d, st / st[0], color=SERIES[1], label="t de Student, 1 g.l. (2D)")
    ax.fill_between(d, g / g[0], st / st[0], where=d > 1.2, color=SERIES[1], alpha=.12)
    ax.text(3.0, 0.35, "cola pesada: los moderadamente\nlejanos pueden alejarse más", fontsize=11, color=INK2)
    ax.set_xlabel("distancia"); ax.set_ylabel("similitud (normalizada)"); ax.legend()
    save(fig, out, "t3_gauss_vs_t")

    # T5: perplexity
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    for ax, perp in zip(axes, [5, 30, 50, 100]):
        Y = TSNE(perplexity=perp, init="pca", random_state=SEED).fit_transform(X)
        scatter_by_size(ax, Y, df, f"perplexity = {perp}")
    save(fig, out, "t5_perplexity")

    # U3: grilla UMAP
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for i, nn in enumerate([5, 50]):
        for j, md in enumerate([0.0, 0.3, 0.9]):
            Y = run_umap(X, n_neighbors=nn, min_dist=md)
            scatter_by_size(axes[i, j], Y, df, f"n_neighbors={nn}, min_dist={md}")
    save(fig, out, "u3_umap_grid")

    # E3/Z2/Z3: tres mapas + métricas
    t0 = time.perf_counter(); Yp = PCA(2).fit_transform(X); tp = time.perf_counter() - t0
    t0 = time.perf_counter(); Yt = TSNE(perplexity=30, init="pca", random_state=SEED).fit_transform(X); tt_ = time.perf_counter() - t0
    run_umap(X[:50], n_neighbors=10)  # calentar numba antes de medir tiempo
    t0 = time.perf_counter(); Yu = run_umap(X, n_neighbors=15, min_dist=0.1); tu = time.perf_counter() - t0
    res["embed_metrics"] = {}
    for name, Y, secs in [("PCA", Yp, tp), ("t-SNE", Yt, tt_), ("UMAP", Yu, tu)]:
        res["embed_metrics"][name] = {
            "trustworthiness": round(float(trustworthiness(X, Y, n_neighbors=10)), 3),
            "knn_preservation": round(knn_preservation(X, Y, 10), 3),
            "seconds": round(secs, 2),
        }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (name, Y) in zip(axes, [("PCA", Yp), ("t-SNE (perplexity 30)", Yt), ("UMAP (15 vecinos)", Yu)]):
        scatter_by_region(ax, Y, df, name, legend=(ax is axes[1]))
    save(fig, out, "z2_three_maps")
    # Z2 alternativo: coloreado por % retraso
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (name, Y) in zip(axes, [("PCA", Yp), ("t-SNE", Yt), ("UMAP", Yu)]):
        sc_ = ax.scatter(Y[:, 0], Y[:, 1], c=df["late_rate"].clip(0, .4), cmap=BLUES, s=7, linewidths=0)
        ax.set_title(name); clean_ax(ax)
    fig.colorbar(sc_, ax=axes, shrink=.8, label="% retraso (tope 40 %)")
    save(fig, out, "z2_three_maps_late")

    # E4: semillas
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    seeds = [0, 1, 2, 3]
    embs = []
    for ax, s in zip(axes, seeds):
        Y = run_umap(X, seed=s, n_neighbors=15, min_dist=0.1)
        embs.append(Y)
        scatter_by_size(ax, Y, df, f"UMAP, semilla {s}")
    save(fig, out, "e4_seeds")
    res["seed_knn_overlap"] = round(float(np.mean([knn_preservation(embs[0], e, 10) for e in embs[1:]])), 3)

    # E5: k-NN sobre embedding recupera región / retraso alto
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    y_late = (df["late_rate"] > df["late_rate"].median()).astype(int).to_numpy()
    res["task_eval"] = {}
    for name, Y in [("original 8D", X), ("PCA 2D", Yp), ("t-SNE 2D", Yt), ("UMAP 2D", Yu)]:
        acc = cross_val_score(KNeighborsClassifier(15), Y, y_late, cv=5).mean()
        res["task_eval"][name] = round(float(acc), 3)
    return res


# ------------------------------------------------------------------ S6
def s6(df: pd.DataFrame, X: np.ndarray, db: Path) -> dict:
    out, res = OUT6, {}
    rng = np.random.default_rng(SEED)

    # K3: inicialización
    Xb, _ = make_blobs(600, centers=[[0, 0], [4, 0], [0, 4], [4, 4]], cluster_std=.6, random_state=SEED)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    runs = [KMeans(4, init="random", n_init=1, random_state=s).fit(Xb) for s in range(200)]
    bad = max(runs, key=lambda m: m.inertia_)
    res["random_init_bad_share"] = round(float(np.mean([m.inertia_ > 1.05 * min(r.inertia_ for r in runs) for m in runs])), 3)
    good = KMeans(4, init="k-means++", n_init=10, random_state=SEED).fit(Xb)
    for ax, m, title in [(axes[0], bad, "init aleatoria, peor de 200 semillas"), (axes[1], good, "k-means++, n_init=10")]:
        ax.scatter(Xb[:, 0], Xb[:, 1], c=colors_for(m.labels_), s=10, linewidths=0)
        ax.scatter(*m.cluster_centers_.T, marker="X", s=180, c=INK, edgecolors="white", linewidths=1.5)
        ax.set_title(f"{title}\ninercia = {m.inertia_:.0f}"); clean_ax(ax)
    res["init_inertia"] = {"bad": round(bad.inertia_, 1), "kmeans++": round(good.inertia_, 1)}
    save(fig, out, "k3_init")

    # K4: fallos
    Xm, _ = make_moons(500, noise=.06, random_state=SEED)
    Xa, _ = make_blobs(500, centers=3, random_state=170)
    Xa = Xa @ np.array([[0.6, -0.64], [-0.41, 0.85]])
    Xs = np.r_[rng.normal([0, 0], .4, (480, 2)), rng.normal([2.5, 0], .4, (20, 2))]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, data, k, title in [(axes[0], Xm, 2, "lunas (no convexos)"), (axes[1], Xa, 3, "anisotrópicos"),
                               (axes[2], Xs, 2, "tamaños muy distintos")]:
        lab = KMeans(k, n_init=10, random_state=SEED).fit_predict(data)
        ax.scatter(data[:, 0], data[:, 1], c=colors_for(lab), s=9, linewidths=0)
        ax.set_title(title); clean_ax(ax)
    save(fig, out, "k4_failures")

    # K5 + V6: elegir k en Olist
    ks = list(range(2, 9))
    rows, labels_k = [], {}
    for k in ks:
        km = KMeans(k, n_init=10, random_state=SEED).fit(X)
        labels_k[k] = km.labels_
        aris = []
        for b in range(10):
            idx = rng.choice(len(X), int(.8 * len(X)), replace=False)
            m2 = KMeans(k, n_init=10, random_state=100 + b).fit(X[idx])
            aris.append(adjusted_rand_score(km.labels_[idx], m2.labels_))
        rows.append({"k": k, "inertia": round(km.inertia_, 0),
                     "silhouette": round(silhouette_score(X, km.labels_), 3),
                     "davies_bouldin": round(davies_bouldin_score(X, km.labels_), 3),
                     "calinski_harabasz": round(calinski_harabasz_score(X, km.labels_), 0),
                     "ari_mean": round(float(np.mean(aris)), 3), "ari_all": np.round(aris, 3).tolist(),
                     "min_size_pct": round(float(np.bincount(km.labels_).min() / len(X)), 3)})
    res["k_table"] = rows
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, [r["inertia"] for r in rows], color=SERIES[0], marker="o", ms=7)
    ax.set_xlabel("$k$"); ax.set_ylabel("inercia"); ax.set_xticks(ks)
    ax.set_title("Codo en vendedores de Olist")
    save(fig, out, "k5_elbow")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot([r["ari_all"] for r in rows], positions=ks, widths=.5, patch_artist=True,
               boxprops=dict(facecolor="#cde2fb", color=SERIES[0]), medianprops=dict(color=SERIES[1], lw=2),
               whiskerprops=dict(color=SERIES[0]), capprops=dict(color=SERIES[0]), flierprops=dict(markeredgecolor=SERIES[0]))
    ax.set_xlabel("$k$"); ax.set_ylabel("ARI vs partición completa"); ax.set_ylim(0, 1.02)
    ax.set_title("Estabilidad: 10 submuestras del 80 %")
    save(fig, out, "v5_ari_box")

    # K7: K-Means vs Mini-Batch
    sizes = [30_000, 100_000, 300_000, 1_000_000]
    t_full, t_mb, ratio = [], [], []
    for n in sizes:
        Xn, _ = make_blobs(n, n_features=20, centers=30, cluster_std=6, random_state=SEED)
        tf, tm = [], []
        for rep in range(3):
            t0 = time.perf_counter(); f = KMeans(30, n_init=1, random_state=rep).fit(Xn); tf.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); mb = MiniBatchKMeans(30, batch_size=4096, n_init=1, random_state=rep).fit(Xn); tm.append(time.perf_counter() - t0)
        t_full.append(float(np.median(tf))); t_mb.append(float(np.median(tm)))
        ratio.append(mb.inertia_ / f.inertia_ - 1)
    res["minibatch"] = {"n": sizes, "t_kmeans": np.round(t_full, 2).tolist(), "t_minibatch": np.round(t_mb, 2).tolist(),
                        "extra_inertia_pct": np.round(np.array(ratio) * 100, 2).tolist()}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(sizes, t_full, color=SERIES[0], marker="o", label="K-Means")
    axes[0].plot(sizes, t_mb, color=SERIES[1], marker="o", label="Mini-Batch")
    axes[0].set_xscale("log"); axes[0].set_xlabel("$n$"); axes[0].set_ylabel("segundos"); axes[0].legend()
    axes[0].set_title("Tiempo de ajuste (20 dims, $k$=30, mediana de 3)")
    axes[1].bar(range(len(sizes)), np.array(ratio) * 100, color=SERIES[1], width=.6)
    axes[1].set_xticks(range(len(sizes)), [f"{n:,}".replace(",", ".") for n in sizes], rotation=20)
    axes[1].set_ylabel("inercia extra (%)"); axes[1].set_title("Costo en calidad de Mini-Batch")
    save(fig, out, "k7_minibatch")

    # H3: dendrograma (muestra)
    idx = rng.choice(len(X), 300, replace=False)
    Zl = linkage(X[idx], method="ward")
    fig, ax = plt.subplots(figsize=(11, 4.4))
    dendrogram(Zl, ax=ax, no_labels=True, color_threshold=Zl[-4, 2], above_threshold_color=INK2,
               link_color_func=None)
    ax.axhline(Zl[-4, 2] * 1.02, color=SERIES[1], ls="--", lw=1.5)
    ax.text(5, Zl[-4, 2] * 1.06, "corte → 4 grupos", color=INK, fontsize=12)
    ax.set_ylabel("altura de fusión (Ward)"); ax.grid(False)
    ax.set_title("Dendrograma: 300 vendedores de Olist")
    save(fig, out, "h3_dendrogram")

    # D2: k-distance
    k_ = 10
    dist = NearestNeighbors(n_neighbors=k_).fit(X).kneighbors()[0][:, -1]
    dist = np.sort(dist)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(dist, color=SERIES[0])
    ax.set_xlabel("vendedores ordenados"); ax.set_ylabel(f"distancia al {k_}º vecino")
    ax.set_title("Gráfico de $k$-distancia (min_samples = 10)")
    q = float(np.quantile(dist, .95)); res["kdist_q95"] = round(q, 3)
    ax.axhline(q, color=SERIES[1], ls="--", lw=1.5); ax.text(20, q * 1.04, f"percentil 95 ≈ {q:.2f}", color=INK)
    save(fig, out, "d2_kdistance")

    # D3: DBSCAN vs HDBSCAN con densidades distintas
    Xv = np.r_[rng.normal([0, 0], .25, (300, 2)), rng.normal([3, 0], 1.0, (300, 2)), rng.uniform(-2, 6, (40, 2)) * [1, .8]]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, eps in zip(axes[:2], [0.2, 0.6]):
        lab = DBSCAN(eps=eps, min_samples=10).fit_predict(Xv)
        ax.scatter(Xv[:, 0], Xv[:, 1], c=colors_for(lab), s=9, linewidths=0)
        ax.set_title(f"DBSCAN eps={eps}\n{lab.max() + 1} cluster(s), {np.mean(lab < 0):.0%} ruido"); clean_ax(ax)
    h = HDBSCAN(min_cluster_size=25).fit(Xv)
    cols = colors_for(h.labels_)
    axes[2].scatter(Xv[:, 0], Xv[:, 1], c=cols, s=4 + 14 * h.probabilities_, linewidths=0)
    axes[2].set_title(f"HDBSCAN min_cluster_size=25\n{h.labels_.max() + 1} clusters, {np.mean(h.labels_ < 0):.0%} ruido"); clean_ax(axes[2])
    save(fig, out, "d3_hdbscan")

    # D5: grilla 4x4
    datasets = {
        "blobs": make_blobs(400, centers=3, cluster_std=.6, random_state=SEED)[0],
        "lunas": Xm,
        "anisotrópico": Xa,
        "densidad variable": make_blobs(400, centers=[[0, 0], [4, 4], [6, 0]], cluster_std=[.3, 1.4, .5], random_state=SEED)[0],
    }
    algos = {
        "K-Means": lambda d, k: KMeans(k, n_init=10, random_state=SEED).fit_predict(d),
        "Ward": lambda d, k: AgglomerativeClustering(k, linkage="ward").fit_predict(d),
        "DBSCAN": lambda d, k: DBSCAN(eps=.3, min_samples=8).fit_predict(d),
        "HDBSCAN": lambda d, k: HDBSCAN(min_cluster_size=20).fit_predict(d),
    }
    kk = {"blobs": 3, "lunas": 2, "anisotrópico": 3, "densidad variable": 3}
    fig, axes = plt.subplots(4, 4, figsize=(12, 10))
    for i, (dn, d) in enumerate(datasets.items()):
        d = StandardScaler().fit_transform(d)
        for j, (an, f) in enumerate(algos.items()):
            lab = f(d, kk[dn])
            axes[i, j].scatter(d[:, 0], d[:, 1], c=colors_for(lab), s=5, linewidths=0)
            clean_ax(axes[i, j])
            if i == 0:
                axes[i, j].set_title(an, fontsize=15, weight="bold")
            if j == 0:
                axes[i, j].set_ylabel(dn, fontsize=13)
    save(fig, out, "d5_grid")

    # V2: silueta Olist con k elegido
    k_sel = choose_k(rows)
    res["k_selected"] = k_sel
    lab = labels_k[k_sel]
    order = df.assign(c=lab).groupby("c")["revenue"].median().sort_values(ascending=False).index
    lab = np.argsort(order.to_numpy())[lab]  # C0 = mayor venta mediana
    sil = silhouette_samples(X, lab)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    y0 = 0
    for c in range(k_sel):
        v = np.sort(sil[lab == c])
        ax.fill_betweenx(np.arange(y0, y0 + len(v)), 0, v, color=SERIES[c], linewidth=0)
        ax.text(-0.08, y0 + len(v) / 2, f"C{c}", va="center", ha="right", fontsize=12)
        y0 += len(v) + 30
    ax.axvline(sil.mean(), color=INK, ls="--", lw=1.2)
    ax.text(sil.mean() + .01, y0 * .98, f"media = {sil.mean():.2f}", fontsize=12)
    ax.set_xlabel("silueta $s(i)$"); ax.set_yticks([]); ax.set_xlim(-0.3, 0.8)
    ax.set_title(f"Gráfico de silueta, K-Means $k$ = {k_sel}")
    save(fig, out, "v2_silhouette")

    # V4: silueta en lunas
    dm = StandardScaler().fit_transform(Xm)
    lk = KMeans(2, n_init=10, random_state=SEED).fit_predict(dm)
    ld = DBSCAN(eps=.3, min_samples=8).fit_predict(dm)
    m = ld >= 0
    res["moons_silhouette"] = {"kmeans": round(silhouette_score(dm, lk), 3),
                               "dbscan": round(silhouette_score(dm[m], ld[m]), 3),
                               "dbscan_noise_pct": round(float(1 - m.mean()), 3)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(dm[:, 0], dm[:, 1], c=colors_for(lk), s=9, linewidths=0)
    axes[0].set_title(f"K-Means: silueta = {res['moons_silhouette']['kmeans']:.2f}")
    axes[1].scatter(dm[:, 0], dm[:, 1], c=colors_for(ld), s=9, linewidths=0)
    axes[1].set_title(f"DBSCAN: silueta = {res['moons_silhouette']['dbscan']:.2f}")
    for ax in axes:
        clean_ax(ax)
    save(fig, out, "v4_moons")

    # S2: perfiles
    prof = df.assign(cluster=lab).groupby("cluster")[FEATURES].median()
    glob = df[FEATURES].median()
    idx_ = prof / glob
    sizes_ = np.bincount(lab)
    res["profiles_median"] = prof.round(3).to_dict("index")
    res["global_median"] = glob.round(3).to_dict()
    res["cluster_sizes"] = sizes_.tolist()
    fig, ax = plt.subplots(figsize=(11, 0.9 + 0.75 * k_sel))
    logi = np.log2(idx_.clip(lower=1 / 8, upper=8).to_numpy(dtype=float))
    im = ax.imshow(logi, cmap=DIVERGING, vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(FEATURES)), [LABELS_ES[f] for f in FEATURES], rotation=25, ha="right")
    ax.set_yticks(range(k_sel), [f"C{c} (n={sizes_[c]})" for c in range(k_sel)])
    for i in range(k_sel):
        for j, f in enumerate(FEATURES):
            ax.text(j, i, fmt_val(f, prof.iloc[i][f]), ha="center", va="center", fontsize=10.5, color=INK)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=.9)
    cb.set_ticks([-2, -1, 0, 1, 2], labels=["¼×", "½×", "= global", "2×", "4×"])
    ax.set_title("Mediana por cluster (color = cluster / mediana global)")
    save(fig, out, "s2_profiles")

    # O2: IF vs LOF
    Xo = np.r_[rng.normal([0, 0], .3, (200, 2)), rng.normal([4, 4], 1.5, (200, 2)), [[1.2, 1.2], [8, 9], [-2, 6]]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    s_if = -IsolationForest(random_state=SEED).fit(Xo).score_samples(Xo)
    lof = LocalOutlierFactor(n_neighbors=20).fit(Xo)
    s_lof = -lof.negative_outlier_factor_
    for ax, s, title in [(axes[0], s_if, "Isolation Forest"), (axes[1], s_lof, "LOF")]:
        top = np.argsort(s)[-6:]
        ax.scatter(Xo[:, 0], Xo[:, 1], s=10, c=SERIES[0], alpha=.5, linewidths=0)
        ax.scatter(Xo[top, 0], Xo[top, 1], s=120, facecolors="none", edgecolors=SERIES[1], linewidths=2)
        ax.set_title(f"{title}: 6 puntos más anómalos"); clean_ax(ax)
    axes[1].annotate("(1.2, 1.2): raro solo\npara su vecindario", xy=(1.2, 1.2), xytext=(4.5, -1.5), fontsize=11,
                     arrowprops=dict(arrowstyle="->", color=INK2))
    save(fig, out, "o2_if_vs_lof")
    res["lof_local_point"] = round(float(s_lof[400]), 2)
    res["if_rank_local_point"] = int((s_if > s_if[400]).sum()) + 1

    # O3: anomalías en vendedores Olist
    iso = IsolationForest(contamination=.01, random_state=SEED).fit(X)
    flag = iso.predict(X) < 0
    lofo = LocalOutlierFactor(n_neighbors=20, contamination=.01).fit_predict(X) < 0
    res["olist_outliers"] = {"if": int(flag.sum()), "lof": int(lofo.sum()), "both": int((flag & lofo).sum()),
                             "if_median": df.loc[flag, FEATURES].median().round(3).to_dict()}

    # R2: drift 2017 vs 2018H1
    con = duckdb.connect(str(db), read_only=True)
    od = con.sql("""
        SELECT order_purchase_timestamp AS ts,
               date_diff('hour', order_purchase_timestamp, order_delivered_customer_date) / 24.0 AS days,
               date_diff('hour', order_purchase_timestamp, order_estimated_delivery_date) / 24.0 AS promised,
               (order_delivered_customer_date > order_estimated_delivery_date)::INT AS late
        FROM orders o
        WHERE order_status = 'delivered' AND order_delivered_customer_date IS NOT NULL
          AND order_purchase_timestamp >= TIMESTAMP '2017-01-01' AND order_purchase_timestamp < TIMESTAMP '2018-09-01'
    """).df()
    con.close()
    od["promised"] = od["promised"].astype(float)
    windows = {
        "days": (od[(od.ts >= "2017-05-01") & (od.ts < "2017-11-01")], od[(od.ts >= "2018-02-01") & (od.ts < "2018-04-01")],
                 "días de entrega", np.r_[0, 5, 10, 15, 20, 30, 45, 200], "may–oct 2017", "feb–mar 2018"),
        "promised": (od[(od.ts >= "2017-05-01") & (od.ts < "2017-11-01")], od[(od.ts >= "2018-07-01") & (od.ts < "2018-09-01")],
                     "plazo prometido (días)", np.r_[0, 10, 15, 20, 25, 30, 40, 200], "may–oct 2017", "jul–ago 2018"),
    }
    res["drift"] = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (col, (ref, cur, lab_, bins, rl, cl)) in zip(axes, windows.items()):
        a = ref[col].dropna(); b = cur[col].dropna()
        e = np.histogram(a, bins)[0] / len(a); c_ = np.histogram(b, bins)[0] / len(b)
        e, c_ = np.clip(e, 1e-4, None), np.clip(c_, 1e-4, None)
        psi = float(np.sum((c_ - e) * np.log(c_ / e)))
        ks = ks_2samp(a, b)
        res["drift"][col] = {"psi": round(psi, 3), "ks": round(float(ks.statistic), 3), "ref": rl, "cur": cl,
                             "ref_median": round(float(a.median()), 2), "cur_median": round(float(b.median()), 2)}
        x = np.arange(len(e))
        ax.bar(x - .2, e, .4, color=SERIES[0], label=f"{rl} (referencia)")
        ax.bar(x + .2, c_, .4, color=SERIES[1], label=f"{cl} (actual)")
        ax.set_xticks(x, [f"{bins[i]:g}–{bins[i+1]:g}" for i in range(len(e))], rotation=25, fontsize=10)
        ax.set_xlabel(lab_); ax.set_ylabel("proporción"); ax.legend(fontsize=10)
        ax.set_title(f"PSI = {psi:.2f} · KS = {ks.statistic:.2f}")
    save(fig, out, "r2_drift")
    mon = od.assign(m=od.ts.dt.to_period("M").dt.to_timestamp()).groupby("m")["late"].mean()
    res["late_by_month"] = {str(k.date()): round(float(v), 3) for k, v in mon.items()}
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(mon.index, mon.values, color=SERIES[0], marker="o", ms=6)
    ax.set_ylabel("% órdenes con retraso"); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_title("El mismo modelo de retrasos enfrenta otro mundo cada mes")
    save(fig, out, "r1_late_monthly")
    return res


def choose_k(rows) -> int:
    """k con mejor silueta entre los que son estables (ARI >= 0.8) y sin clusters < 5 %."""
    ok = [r for r in rows if r["ari_mean"] >= .8 and r["min_size_pct"] >= .05 and r["k"] >= 3]
    ok = ok or rows
    return max(ok, key=lambda r: r["silhouette"])["k"]


def fmt_val(f, v):
    if f in ("late_rate", "freight_ratio"):
        return f"{v:.0%}"
    if f == "revenue":
        return f"{v/1000:.1f}k"
    if f == "review_mean":
        return f"{v:.2f}"
    return f"{v:.0f}" if v >= 10 else f"{v:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, required=True, help="archivo olist.duckdb")
    parser.add_argument("--only", choices=["s5", "s6"], default=None)
    args = parser.parse_args()

    df = load_sellers(args.db)
    X = prepare(df)
    print(f"{len(df)} vendedores con >= 3 órdenes entregadas antes de 2018-06-01")
    if args.only in (None, "s5"):
        r = s5(df, X)
        (OUT5 / "results_s5.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))
    if args.only in (None, "s6"):
        r = s6(df, X, args.db)
        (OUT6 / "results_s6.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
