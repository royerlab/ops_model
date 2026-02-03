# %%
import pandas as pd
import zarr
from ops_model.data.paths import OpsPaths

labels_df = pd.read_csv(
    "/hpc/mydata/alexander.hillsley/ops/ops_model/experiments/models/dynaclr/labels_testset.csv",
    low_memory=False,
)

# Find the one where the channel =='Cy5' and check the channel exists in the store
cy5_rows = labels_df[labels_df["channel"] == "Cy5"]
print(f"Number of Cy5 rows: {len(cy5_rows)}")
print(f"Experiments with Cy5: {cy5_rows['store_key'].unique()}")

# %%
# Check the channel exists in the store for each unique (store_key, well) combination
unique_combos = cy5_rows[["store_key", "well"]].drop_duplicates()
print(f"Unique (store_key, well) combinations with Cy5: {len(unique_combos)}")

stores_cache = {}
missing_cy5 = []

for _, row in unique_combos.iterrows():
    store_key = row["store_key"]
    well = row["well"]

    if store_key not in stores_cache:
        v3_path = OpsPaths(store_key).stores["phenotyping_v3"]
        stores_cache[store_key] = zarr.open_group(v3_path, mode="r")

    store = stores_cache[store_key]
    img = store[well]
    attrs = img.attrs.asdict()
    channels = [a["label"] for a in attrs["ome"]["omero"]["channels"]]

    if "Cy5" not in channels:
        missing_cy5.append((store_key, well, channels))

print(f"\nWells missing Cy5 channel: {len(missing_cy5)}")
for store_key, well, channels in missing_cy5[:10]:
    print(f"  {store_key} | {well} | channels: {channels}")

if len(missing_cy5) > 10:
    print(f"  ... and {len(missing_cy5) - 10} more")

# %%
