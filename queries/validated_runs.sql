SELECT *
FROM read_csv_auto(
  'analysis/report-data/validated_runs.csv',
  header = true
);
