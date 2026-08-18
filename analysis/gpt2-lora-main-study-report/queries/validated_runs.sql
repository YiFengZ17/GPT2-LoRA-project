-- DuckDB: reviewed per-run experiment metrics.
SELECT *
FROM read_csv_auto('data/validated_runs.csv', header = true);
