-- DuckDB: data-quality and cross-file consistency checks.
SELECT *
FROM read_csv_auto('data/quality_checks.csv', header = true);
