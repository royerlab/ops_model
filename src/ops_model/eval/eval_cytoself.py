import yaml
from pathlib import Path

import pandas as pd
import torch
import lightning as L

from ops_model.data import data_loader
from ops_model.models import cytoself_model

torch.multiprocessing.set_sharing_strategy("file_system")


def run_inference(
    config_path: str,
    checkpoint_path: str,
    output_path: str,
):
    output_path = Path(output_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_type = config["model_type"]
    dataset_type = config["dataset_type"]

    data_manager = data_loader.OpsDataManager(
        experiments=config["data_manager"]["experiments"],
        batch_size=config["data_manager"]["batch_size"],
        data_split=(0, 0, 1),
        out_channels=config["data_manager"]["out_channels"],
        initial_yx_patch_size=tuple(config["data_manager"]["initial_yx_patch_size"]),
        final_yx_patch_size=tuple(config["data_manager"]["final_yx_patch_size"]),
    )
    data_manager.construct_dataloaders(
        num_workers=config["data_manager"]["num_workers"],
        dataset_type=dataset_type,
        basic_kwargs=config["data_manager"].get("basic_kwargs"),
        triplet_kwargs=config["data_manager"].get("triplet_kwargs"),
    )

    test_loader = data_manager.test_loader

    torch.set_float32_matmul_precision("medium")  # huge boost in speed

    mc = config["model"]
    lit_model = cytoself_model.LitCytoSelf.load_from_checkpoint(
        checkpoint_path,
        emb_shapes=(
            tuple(mc["embedding_shapes"][0]),
            tuple(mc["embedding_shapes"][1]),
        ),
        vq_args=mc["vq_args"],
        num_class=mc["num_classes"],
        input_shape=tuple(mc["input_shape"]),
        output_shape=tuple(mc["input_shape"]),
        fc_input_type=mc["fc_input_type"],
        fc_output_idx=[mc["fc_output_index"]],
    )
    pred_writer = cytoself_model.CytoselfPredictionWriter

    trainer = L.Trainer(
        devices=1,
        accelerator="gpu",
        callbacks=[
            pred_writer(
                output_dir=output_path,
                write_interval="batch",
                int_label_lut=data_manager.int_label_lut,
            )
        ],
        # limit_predict_batches=2
    )
    predictions = trainer.predict(lit_model, dataloaders=test_loader)

    aggregate_csvs(
        chunk_subdir=output_path / "emb_2_chunks",
        final_csv_name="cytoself_local_features.csv",
    )
    aggregate_csvs(
        chunk_subdir=output_path / "classification_scores",
        final_csv_name="cytoself_classification_scores.csv",
    )
    aggregate_csvs(
        chunk_subdir=output_path / "global_emb_metadata",
        final_csv_name="cytoself_global_metadata.csv",
    )

    return


def aggregate_csvs(
    chunk_subdir: Path,
    final_csv_name: str,
):
    print(f"\nLoading and concatenating chunks from {chunk_subdir.name}...")
    csv_files = sorted(chunk_subdir.glob("*.csv"))

    if not csv_files:
        print("No feature files found!")
        return None

    df_list = [pd.read_csv(csv_file) for csv_file in csv_files]
    final_df = pd.concat(df_list, ignore_index=True)

    # Save the final concatenated dataframe
    final_path = chunk_subdir.parent / final_csv_name
    final_df.to_csv(final_path, index=False)
    print(f"Saved final concatenated features to {final_path}")
    print(f"Final dataframe shape: {final_df.shape}")

    return


if __name__ == "__main__":
    checkpoint_path = "/hpc/projects/intracellular_dashboard/ops/models/logs/cytoself/cytoself_20251202_2/cytoself_20251202_2-2025-12-04-global_step=0.000000-val/total_loss=330.02.ckpt"
    config_path = "/hpc/mydata/alexander.hillsley/ops/ops_model/configs/cytoself/cytoself_20251204.yml"
    output_path = "/hpc/projects/intracellular_dashboard/ops/ops0031_20250424/3-assembly/cytoself_features"
    run_inference(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
    )
