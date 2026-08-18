SELECT *
FROM read_csv_auto(
  'analysis/report-data/validated_aggregate.csv',
  header = true
);
