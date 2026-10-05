"""Construye la base de datos DuckDB del taller 2026-2 a partir del dataset Olist.

Descarga "Brazilian E-Commerce Public Dataset by Olist" desde Kaggle (kagglehub)
y carga cada CSV como una tabla en un único archivo .duckdb. Las tablas se
cargan tal cual (sin limpieza): la validación y la limpieza son parte del taller.

Uso:
    uv run python homeworks/taller/build_olist_duckdb.py              # crea ./olist.duckdb
    uv run python homeworks/taller/build_olist_duckdb.py --out data/olist.duckdb
    uv run python homeworks/taller/build_olist_duckdb.py --csv-dir ruta/a/csvs

Fuente: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
Licencia: CC BY-NC-SA 4.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

KAGGLE_HANDLE = "olistbr/brazilian-ecommerce"

# tabla -> archivo CSV original
TABLES = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

# columnas que deben quedar como TIMESTAMP
TIMESTAMPS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_items": ["shipping_limit_date"],
    "order_reviews": ["review_creation_date", "review_answer_timestamp"],
}

# columnas que deben quedar como texto (códigos postales con ceros a la izquierda)
VARCHARS = {
    "customers": ["customer_zip_code_prefix"],
    "sellers": ["seller_zip_code_prefix"],
    "geolocation": ["geolocation_zip_code_prefix"],
}


def download_csvs() -> Path:
    import kagglehub

    return Path(kagglehub.dataset_download(KAGGLE_HANDLE))


def build(csv_dir: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    con = duckdb.connect(str(out))
    for table, filename in TABLES.items():
        types = {c: "TIMESTAMP" for c in TIMESTAMPS.get(table, [])}
        types |= {c: "VARCHAR" for c in VARCHARS.get(table, [])}
        if types:
            sql, params = "read_csv(?, header = true, types = ?)", [str(csv_dir / filename), types]
        else:
            sql, params = "read_csv(?, header = true)", [str(csv_dir / filename)]
        con.execute(f"CREATE TABLE {table} AS SELECT * FROM {sql}", params)
        n = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        print(f"  {table:<22}{n:>10,} filas")
    con.close()
    print(f"\nBase creada en {out.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("olist.duckdb"), help="archivo .duckdb de salida")
    parser.add_argument("--csv-dir", type=Path, default=None, help="carpeta con los CSV (omite la descarga)")
    args = parser.parse_args()

    csv_dir = args.csv_dir or download_csvs()
    print(f"CSV en {csv_dir}")
    build(csv_dir, args.out)


if __name__ == "__main__":
    main()
