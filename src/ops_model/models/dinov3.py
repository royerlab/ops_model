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


class DinoV3Model:
    """DINOv3 ViT-L/16 (Meta, natural-image pretrained) inference wrapper.

    Loads the model from a local torch.hub checkout and returns one embedding
    per image via ``model.forward``. Unlike Cell-DINO, preprocessing uses fixed
    ImageNet-style channel statistics rather than per-image z-scoring.
    """

    def __init__(self):
        super().__init__()
        self.load_model()
        self.model.cuda()

    def load_model(self):
        repo_dir = OpsPaths.checkpoint("dinov3", "dinov3")
        checkpoint = OpsPaths.checkpoint(
            "dinov3", "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        )

        self.model = torch.hub.load(
            str(repo_dir), "dinov3_vitl16", source="local", weights=str(checkpoint)
        )
        return

    def extract_features(self, batch: dict) -> torch.Tensor:

        x = batch["data"]
        inputs = self.preprocess(x)

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                output = self.model(inputs.cuda())
        return output

    def preprocess(self, x: torch.Tensor, resize_size: int = 256) -> torch.Tensor:
        """Resize, scale to [0, 1], and divide by ImageNet channel std.

        DINOv3 expects roughly unit-variance inputs. We scale to [0, 1] then
        divide by the ImageNet std; mean is left at 0 (no centering), matching
        the original DINOv3 preprocessing for these grayscale-derived crops.
        """
        to_tensor = v2.ToImage()
        resize = v2.Resize((resize_size, resize_size), antialias=True)
        to_float = v2.ToDtype(torch.float32, scale=True)
        normalize = v2.Normalize(
            mean=(0, 0, 0),
            std=(0.229, 0.224, 0.225),
        )
        return v2.Compose([to_tensor, resize, to_float, normalize])(x.cuda())


def extract_dinov3_features(config: dict = None):
    print(
        f"Extracting DINOv3 features for {list(config['data_manager']['experiments'].keys())}"
    )
    out_channels = config["data_manager"]["out_channels"]
    dm = build_dataloader(
        config, out_channels, labels_df=maybe_build_labels_df(config)
    )
    model = DinoV3Model()
    run_extraction(
        config, dm, model, model_prefix="dinov3", name_channel=out_channels[0]
    )


def dinov3_main(config_paths: list, run_anndata: bool = True):
    """
    Main orchestrator function for DinoV3 feature extraction.

    Spawns one SLURM job per config × channel via submit_parallel_jobs.
    Each job runs extract_dinov3_features() independently. Once extraction
    completes, optionally submits the AnnData conversion
    (batch_process_embeddings) as a follow-on CPU SLURM batch.

    Args:
        config_paths: List of paths to YAML configuration files
        run_anndata: If True (default), convert the extracted feature CSVs into
            AnnData objects after extraction succeeds.

    Returns:
        Result dict from submit_parallel_jobs (extraction)
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
                    "func": extract_dinov3_features,
                    "kwargs": {"config": channel_config},
                    "metadata": {
                        "config": config_path,
                        "channel": channel,
                        "experiments": experiments,
                    },
                }
            )

    log_dir = str(OpsPaths.slurm_log_dir("dinov3"))

    print(f"\nSubmitting {len(jobs_to_submit)} DinoV3 job(s) via submit_parallel_jobs")
    result = submit_parallel_jobs(
        jobs_to_submit=jobs_to_submit,
        experiment="dinov3",
        slurm_params=slurm_params,
        log_dir=log_dir,
        wait_for_completion=True,
    )

    if run_anndata:
        run_anndata_followon(config_paths, result)

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract DINOv3 features from OPS dataset based on config"
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
    dinov3_main(config_paths, run_anndata=not args.skip_anndata)
