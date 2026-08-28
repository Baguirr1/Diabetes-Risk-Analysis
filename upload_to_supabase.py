"""
One-off script: uploads diabetes_risk.csv into a Postgres table on Supabase.

Usage:
    export DATABASE_URL="postgresql://postgres.xxxx:PASSWORD@aws-0-region.pooler.supabase.com:6543/postgres"
    python upload_to_supabase.py
"""

import os
import sys

import pandas as pd
from sqlalchemy import create_engine

TABLE_NAME = "diabetes_risk"


def main():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        sys.exit("Set the DATABASE_URL environment variable to your Supabase connection string first.")

    df = pd.read_csv("diabetes_risk.csv")
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns from diabetes_risk.csv")

    engine = create_engine(database_url)
    df.to_sql(TABLE_NAME, engine, if_exists="replace", index=False, chunksize=1000, method="multi")
    print(f"Uploaded to table '{TABLE_NAME}' on Supabase.")

    with engine.connect() as conn:
        count = conn.exec_driver_sql(f"SELECT COUNT(*) FROM {TABLE_NAME}").scalar()
        print(f"Row count in Supabase table: {count:,}")


if __name__ == "__main__":
    main()
