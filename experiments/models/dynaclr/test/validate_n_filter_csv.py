"""
Validate that channels referenced in a labels CSV actually exist in the
corresponding zarr stores.

Prints a detailed diagnosis of mismatches, then writes a filtered CSV
with only valid rows.

Usage
-----
    python validate_n_filter_csv.py
"""

# %%
import pandas as pd
import zarr

from ops_model.data.paths import OpsPaths

# %%
INPUT_PATH = "/home/eduardo.hirata/repos/ops_model/experiments/models/dynaclr/test/labels_testset_10complex_n_NTC_v2.csv"
OUTPUT_PATH = "/home/eduardo.hirata/repos/ops_model/experiments/models/dynaclr/test/labels_testset_10complex_n_NTC_v2_filtered.parquet"

# %%
if INPUT_PATH.endswith(".parquet"):
    df = pd.read_parquet(INPUT_PATH)
else:
    df = pd.read_csv(INPUT_PATH, low_memory=False)
print("## Input\n")
print(f"- **File:** `{INPUT_PATH}`")
print(f"- **Rows:** {len(df):,}")
print(f"- **Unique store_keys:** {df['store_key'].nunique()}")
print(f"- **Unique channels in CSV:** {sorted(df['channel'].unique())}")

# %%
# Open zarr stores and build channel lookup per (store_key, well)
stores_cache = {}
channel_cache = {}

for store_key in df["store_key"].unique():
    if store_key not in stores_cache:
        stores_cache[store_key] = zarr.open_group(
            OpsPaths(store_key).stores["phenotyping_v3"], mode="r"
        )

    store = stores_cache[store_key]
    for well in df[df["store_key"] == store_key]["well"].unique():
        try:
            attrs = store[well].attrs.asdict()
            zarr_channels = {a["label"] for a in attrs["ome"]["omero"]["channels"]}
        except (KeyError, TypeError):
            zarr_channels = set()
        channel_cache[(store_key, well)] = zarr_channels

# %%
# Diagnose: show every (store_key, well) where a CSV channel is missing
print("\n## Channel diagnosis\n")

mismatches = []
for (store_key, well), zarr_channels in sorted(channel_cache.items()):
    subset = df[(df["store_key"] == store_key) & (df["well"] == well)]
    csv_channels = set(subset["channel"].unique())
    missing = csv_channels - zarr_channels

    if missing:
        n_affected = len(subset[subset["channel"].isin(missing)])
        mismatches.append(
            {
                "store_key": store_key,
                "well": well,
                "missing_channels": sorted(missing),
                "zarr_channels": sorted(zarr_channels),
                "affected_rows": n_affected,
            }
        )

if mismatches:
    n = len(mismatches)
    print(f"Found **{n}** (store_key, well) combos with missing channels:\n")
    for m in mismatches:
        print(
            f"- `{m['store_key']}` / `{m['well']}`: "
            f"CSV references **{m['missing_channels']}** "
            f"but zarr only has `{m['zarr_channels']}` "
            f"({m['affected_rows']:,} rows)"
        )
else:
    print("All CSV channels exist in their zarr stores.")

# %%
# Filter invalid rows
valid_mask = df.apply(
    lambda row: (
        row["channel"] in channel_cache.get((row["store_key"], row["well"]), set())
    ),
    axis=1,
)

bad_df = df[~valid_mask]
n_removed = len(bad_df)

print("\n## Filtering summary\n")
pct = 100 * n_removed / len(df)
print(f"- **Rows removed:** {n_removed:,} / {len(df):,} ({pct:.2f}%)")

if n_removed > 0:
    summary = bad_df.groupby(["store_key", "channel"]).size().reset_index(name="count")
    print(f"\n{summary.to_markdown(index=False)}")

# %%
df_clean = df[valid_mask].reset_index(drop=True)
print("\n## Output\n")
print(f"- **Rows:** {len(df_clean):,}")
print(f"- **Saved to:** `{OUTPUT_PATH}`")

if OUTPUT_PATH.endswith(".parquet"):
    df_clean.to_parquet(OUTPUT_PATH, index=False)
else:
    df_clean.to_csv(OUTPUT_PATH, index=False)

# %%
