"""OLAP-style questions over the lake with DuckDB (Milestone 2).

DuckDB reads the partitioned Parquet lake in place -- no database server, no load step.
Notice how fast column-subset queries are: Parquet is column-major (Ch. 3).
"""

from __future__ import annotations

import duckdb

from taxieta import config

GLOB = str(config.LAKE_DIR / "*" / "*" / "*.parquet")
SRC = f"read_parquet('{GLOB}', hive_partitioning=1)"

QUERIES = {
    "duration_by_hour_of_week": f"""
        SELECT pickup_dow, pickup_hour,
               round(median(duration_s)/60, 1) AS median_min, count(*) AS n
        FROM {SRC}
        GROUP BY ALL ORDER BY pickup_dow, pickup_hour""",
    "busiest_pickup_zones": f"""
        SELECT pu_zone, any_value(pu_borough) AS borough, count(*) AS n
        FROM {SRC} GROUP BY pu_zone ORDER BY n DESC LIMIT 10""",
    "airport_share": f"""
        SELECT year, month,
               round(100.0 * avg(CASE WHEN pu_zone IN {tuple(config.AIRPORT_ZONES)}
                                        OR do_zone IN {tuple(config.AIRPORT_ZONES)} THEN 1 ELSE 0 END), 2)
                 AS airport_pct,
               count(*) AS trips
        FROM {SRC} GROUP BY ALL ORDER BY year, month""",
    "monthly_speed": f"""
        SELECT year, month, round(median(trip_distance / (duration_s/3600)), 2) AS median_mph
        FROM {SRC} WHERE duration_s > 60 GROUP BY ALL ORDER BY year, month""",
    # SOLVED(M2): extra questions
    "most_variable_borough_pairs": f"""
        SELECT pu_borough, do_borough, count(*) AS n,
               round(median(duration_s)/60, 1) AS median_min,
               round((quantile_cont(duration_s, 0.9) - quantile_cont(duration_s, 0.1))/60, 1) AS p10_p90_spread_min
        FROM {SRC} GROUP BY ALL HAVING count(*) >= 100
        ORDER BY p10_p90_spread_min DESC LIMIT 10""",
    "rush_vs_night_speed": f"""
        SELECT CASE WHEN pickup_hour IN (7,8,16,17) THEN 'rush' WHEN pickup_hour BETWEEN 0 AND 5
                    THEN 'night' ELSE 'other' END AS period,
               round(median(trip_distance / (duration_s/3600)), 2) AS median_mph, count(*) AS n
        FROM {SRC} WHERE duration_s > 60 AND trip_distance > 0 GROUP BY ALL ORDER BY median_mph""",
}


def run(name: str | None = None) -> None:
    names = [name] if name else list(QUERIES)   # if name is provided, run only that query, otherwise run all queries
    for n in names:
        print(f"\n== {n} ==")
        print(duckdb.sql(QUERIES[n]).df().to_string(index=False))
