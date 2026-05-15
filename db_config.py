"""
PostgreSQL connection configuration.

Mirrors the `db_properties` dict pattern from Lecture 3 (SQL & SparkSQL).
The defaults match the course defaults (host=localhost:5432,
user=postgres, password=bigdata, database=postgres).

Override via environment variables if you have a non-default Postgres setup:

    DB_HOST     (default: localhost)
    DB_PORT     (default: 5432)
    DB_USER     (default: postgres)
    DB_PASSWORD (default: bigdata)
    DB_NAME     (default: postgres)
"""

from __future__ import annotations

import os
from typing import Any


def db_properties() -> dict[str, Any]:
    """Return the connection properties dict used by SparkSQL + psycopg2."""
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    name = os.environ.get("DB_NAME", "postgres")
    return {
        "host": host,
        "port": port,
        "database": name,
        "username": os.environ.get("DB_USER", "postgres"),
        "password": os.environ.get("DB_PASSWORD", "bigdata"),
        "url": f"jdbc:postgresql://{host}:{port}/{name}",
        "driver": "org.postgresql.Driver",
    }


def psycopg2_dsn() -> dict[str, Any]:
    """Return kwargs suitable for psycopg2.connect()."""
    p = db_properties()
    return {
        "host": p["host"],
        "port": int(p["port"]),
        "dbname": p["database"],
        "user": p["username"],
        "password": p["password"],
    }
