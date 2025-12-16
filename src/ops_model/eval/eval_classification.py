import yaml

import pandas as pd
import torch

from ops_model.data import data_loader

NONFEATURE_COLUMNS = [
    "label_str",
    "label_int",
    "sgRNA",
    "well",
    "experiment",
    "x_position",
    "y_position",
]


def eval_classification_accuracy(
    scores_df: pd.DataFrame, labels_df: pd.DataFrame, label_column: str = None
) -> dict:
    """
    Evaluate classification accuracy given prediction scores and true labels.

    Args:
        scores_df: DataFrame with classification scores, shape (n_samples, n_classes).
        labels_df: DataFrame containing the true integer labels.
        label_column: Name of the column in labels_df containing the labels.

    Returns:
        dict: Dictionary containing accuracy metrics.
    """
    # Convert to tensors
    scores = torch.from_numpy(scores_df.values)

    # Extract true labels
    if label_column is not None:
        labels = torch.from_numpy(labels_df[label_column].values)
    elif len(labels_df.columns) == 1:
        labels = torch.from_numpy(labels_df.iloc[:, 0].values)
    else:
        labels = torch.from_numpy(labels_df.values.flatten())

    # Get predictions and calculate accuracy
    predictions = torch.argmax(scores, dim=1)
    accuracy = (predictions == labels).float().mean().item()

    return {
        "accuracy": accuracy,
        "correct": (predictions == labels).sum().item(),
        "total": len(labels),
        "predictions": predictions.numpy(),
    }

    return


def cnn_inference(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    experiment_dict = {
        "ops0031_20250424": ["A/1/0", "A/2/0", "A/3/0"],
        # "ops0053_20250709": ["A/1/0", "A/2/0", "A/3/0"],
        # "ops0079_20250916": ["A/1/0", "A/2/0", "A/3/0"],
        # "ops0064_20250811": ["A/1/0", "A/2/0", "A/3/0"],
        # "ops0065_20250812": ["A/1/0", "A/2/0", "A/3/0"],
    }
    run_name = config["run_name"]

    dm = data_loader.OpsDataManager(
        experiments=experiment_dict,
        batch_size=config["data_manager"]["batch_size"],
        data_split=tuple(config["data_manager"]["data_split"]),
        out_channels=config["data_manager"]["out_channels"],
        initial_yx_patch_size=tuple(config["data_manager"]["initial_yx_patch_size"]),
        final_yx_patch_size=tuple(config["data_manager"]["final_yx_patch_size"]),
        verbose=False,
    )

    # Construct dataloaders first (without sampler) to get the train indices
    dm.construct_dataloaders(
        num_workers=config["data_manager"]["num_workers"],
        dataset_type=config["dataset_type"],
        basic_kwargs=config["data_manager"].get("basic_kwargs"),
        balanced_sampling=config["data_manager"].get("balanced_sampling", False),
    )
    return


if __name__ == "__main__":
    save_path = "/hpc/projects/intracellular_dashboard/ops/ops0031_20250424/3-assembly/cytoself_features/cytoself_classification_scores.csv"
    df = pd.read_csv(save_path)
    labels_df = df[["label_int"]]
    classification_scores = df.drop(columns=NONFEATURE_COLUMNS)

    a = eval_classification_accuracy(
        scores_df=classification_scores, labels_df=labels_df, label_column="label_int"
    )
