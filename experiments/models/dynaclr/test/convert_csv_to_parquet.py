"""Convert a labels CSV file to Parquet format."""

# %%
from ops_model.data.qc.qc_labels import csv_to_parquet

csv_path = "/home/eduardo.hirata/repos/ops_model/experiments/models/dynaclr/test/labels_testset_10complex_n_NTC_v2_filtered.csv"
parquet_path = "/home/eduardo.hirata/repos/ops_model/experiments/models/dynaclr/test/labels_testset_10complex_n_NTC_v2_filtered.parquet"

csv_to_parquet(csv_path, parquet_path)

# %%
