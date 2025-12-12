from ast import literal_eval

import pandas as pd


def filter_small_bboxes(
    df: pd.DataFrame,
    threshold: int = 5,
) -> pd.DataFrame:

    def bbox_y_length(s):
        t = literal_eval(s)
        return (t[2] - t[0]) > threshold

    def bbox_x_length(s):
        t = literal_eval(s)
        return (t[3] - t[1]) > threshold

    y_pass = df["bbox"].apply(bbox_y_length)
    x_pass = df["bbox"].apply(bbox_x_length)
    length_pass = y_pass & x_pass
    filtered_df = df[length_pass]
    num_cells_removed = len(df) - len(filtered_df)

    return filtered_df, num_cells_removed
