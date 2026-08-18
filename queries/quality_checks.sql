SELECT *
FROM read_csv_auto(
  'analysis/report-data/quality_checks.csv',
  header = true
);
