-- DuckDB: reviewed configuration-level experiment metrics.
SELECT *
FROM read_csv_auto('data/validated_aggregate.csv', header = true);
