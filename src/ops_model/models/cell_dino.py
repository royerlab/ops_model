import argparse
import copy
from pathlib import Path

import yaml
import torch
from torchvision.transforms import v2

from ops_model.data.paths import OpsPaths
from ops_model.models.extractor_common import (
    build_dataloader,
    build_slurm_params,
    maybe_build_labels_df,
    run_anndata_followon,
    run_extraction,
)

REPO_DIR = str(OpsPaths.checkpoint("cell_dino", "dinov2"))
CHECKPOINT = str(
    OpsPaths.checkpoint(
        "cell_dino", "channel_adaptive_dino_vitl16_pretrain_cells-ef7c17ff.pth"
    )
)


class CellDinoModel:

    def __init__(self, z_score: bool = True):
        super().__init__()
        self.z_score = z_score
        self.load_model()
        self.model.cuda()

    def load_model(self):
        self.model = torch.hub.load(
            REPO_DIR,
            "channel_adaptive_dino_vitl16",
            source="local",
            pretrained_path=CHECKPOINT,
            in_channels=1,
        )
        return

    def extract_features(self, batch: dict) -> torch.Tensor:

        x = batch["data"]
        inputs = self.preprocess(x)

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                output = self.model(inputs.cuda())
        return output

    def preprocess(self, x: torch.Tensor, resize_size: int = 224) -> torch.Tensor:
        """Resize to ``resize_size`` and optionally per-image z-score normalize.

        Unlike DINOv3, Cell-DINO uses per-image z-score rather than global ImageNet
        statistics. Input is expected to be a float tensor (our dataloader already
        produces floats), so no uint8->float conversion is needed.

        When ``self.z_score`` is False the resized image is returned without
        z-scoring (used by the "only per-well normalization" sweep condition,
        where normalization is handled entirely in the dataloader).
        """
        resize = v2.Resize((resize_size, resize_size), antialias=True)
        x = resize(x.cuda().float())
        if not self.z_score:
            return x
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True)
        return (x - mean) / (std + 1e-7)


def extract_cell_dino_features(config: dict = None):
    print(
        f"Extracting Cell-DINO features for {list(config['data_manager']['experiments'].keys())}"
    )
    out_channels = config["data_manager"]["out_channels"]
    dm = build_dataloader(
        config, out_channels, labels_df=maybe_build_labels_df(config)
    )
    model = CellDinoModel(z_score=config["data_manager"].get("model_z_score", True))
    run_extraction(
        config, dm, model, model_prefix="cell_dino", name_channel=out_channels[0]
    )


def cell_dino_main(config_paths: list[str], run_anndata: bool = True):
    """Orchestrate Cell-DINO feature extraction via SLURM.

    For each config, reads out_channels and spawns one SLURM job per channel,
    using submit_parallel_jobs. Once extraction completes, optionally submits the
    AnnData conversion (batch_process_embeddings) as a follow-on CPU SLURM batch.

    Args:
        config_paths: List of paths to YAML configuration files.
        run_anndata: If True (default), convert the extracted feature CSVs into
            AnnData objects after extraction succeeds.

    Returns:
        Result dict from submit_parallel_jobs (extraction).
    """
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

    jobs_to_submit = []
    slurm_params = None

    for config_path in config_paths:
        config_stem = Path(config_path).stem
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        out_channels = config["data_manager"]["out_channels"]
        experiments = list(config["data_manager"]["experiments"].keys())

        print(
            f"Config {config_stem}: {len(out_channels)} channel(s) for experiment(s) {experiments}"
        )
        print(f"  Channels: {out_channels}")

        if slurm_params is None:
            slurm_params = build_slurm_params(config, default_cpus=20)

        for channel in out_channels:
            channel_config = copy.deepcopy(config)
            channel_config["data_manager"]["out_channels"] = [channel]
            jobs_to_submit.append(
                {
                    "name": f"{config_stem}_{channel}",
                    "func": extract_cell_dino_features,
                    "kwargs": {"config": channel_config},
                    "metadata": {
                        "config": config_path,
                        "channel": channel,
                        "experiments": experiments,
                    },
                }
            )

    log_dir = str(OpsPaths.slurm_log_dir("cell_dino"))

    print(
        f"\nSubmitting {len(jobs_to_submit)} Cell-DINO job(s) via submit_parallel_jobs"
    )
    result = submit_parallel_jobs(
        jobs_to_submit=jobs_to_submit,
        experiment="cell_dino",
        slurm_params=slurm_params,
        log_dir=log_dir,
        wait_for_completion=True,
    )

    if run_anndata:
        run_anndata_followon(config_paths, result)

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract Cell-DINO features from OPS dataset based on config"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config_path",
        type=str,
        help="Path to a single YAML config file",
    )
    group.add_argument(
        "--config_list",
        type=str,
        help="Path to .txt file with one config path per line",
    )
    parser.add_argument(
        "--skip_anndata",
        action="store_true",
        help="Skip the automatic AnnData conversion after feature extraction",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.config_list:
        with open(args.config_list) as f:
            config_paths = [line.strip() for line in f if line.strip()]
    else:
        config_paths = [args.config_path]
    cell_dino_main(config_paths, run_anndata=not args.skip_anndata)
