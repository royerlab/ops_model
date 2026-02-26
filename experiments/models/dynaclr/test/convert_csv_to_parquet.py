"""Convert a labels CSV file to Parquet format."""

# %%
from ops_model.data.qc.qc_labels import csv_to_parquet

csv_path = "/home/eduardo.hirata/repos/ops_model/experiments/models/dynaclr/test/labels_testset_10complex_n_NTC_v2_filtered.csv"
parquet_path = "/home/eduardo.hirata/repos/ops_model/experiments/models/dynaclr/test/labels_testset_10complex_n_NTC_v2_filtered.parquet"

csv_to_parquet(csv_path, parquet_path)

# %%
import pandas as pd


# FIlter the parquet file to only include Phase 2D data
parquet_path = "/home/eduardo.hirata/repos/ops_model/experiments/models/dynaclr/test/labels_testset_10complex_n_NTC_v2_filtered.parquet"
output_path = "/home/eduardo.hirata/repos/ops_model/experiments/models/dynaclr/test/labels_testset_10complex_n_NTC_v2_filtered_phase2d.parquet"
df = pd.read_parquet(parquet_path)
df = df[df["channel"] == "Phase2D"]
df.to_parquet(output_path)
