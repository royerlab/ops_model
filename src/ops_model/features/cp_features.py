# %%
from joblib import Parallel, delayed

from tqdm import tqdm
import numpy as np
import pandas as pd
from cp_measure.bulk import get_core_measurements
import torch

from ops_model.data import data_loader
from ops_model.data.paths import OpsPaths

torch.multiprocessing.set_sharing_strategy("file_system")


def single_channel_features(
    img: np.ndarray,
    mask: np.ndarray,
    measurements: dict = None,
    prefix: str = "",
):
    """
    cp_measure requires:
    img: H, W (either int of float between -1 and 1)
    mask: H, W (int)
    """

    img = np.squeeze(img).astype(np.uint16)
    mask = np.squeeze(mask).astype(np.uint16)

    results = {}
    for name, fn in measurements.items():
        features = fn(mask, img)
        features_prefixed = {f"{prefix}_{name}": v for name, v in features.items()}
        results.update(features_prefixed)

    return results


def shape_features(
    img: np.ndarray,
    mask: np.ndarray,
    prefix: str = "",
    fcn=None,
):
    """
    cp_measure requires:
    img: H, W (either int of float between -1 and 1)
    mask: H, W (int)
    """

    img = np.squeeze(img).astype(np.uint16)
    mask = np.squeeze(mask).astype(np.uint16)

    results = {}
    features = fcn(mask, img)
    features_prefixed = {
        f"{prefix}_sizeshape_{name}": v for name, v in features.items()
    }
    results.update(features_prefixed)

    return results


def save_cp_features(experiment_dict, features, override: bool = False):

    save_path = OpsPaths(list(experiment_dict.keys())[0]).cell_profiler_out
    if save_path.exists() and not override:
        print(f"file exists at {save_path}, use override=True to overwrite")

    if not save_path.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)

    features.to_csv(save_path, index=False)

    return


def cp_features(
    experiment_dict: dict = None,
    verbose: bool = False,
    num_workers: int = 4,
):
    """
    CAUTION: should only be run for a single experiment at a time
    """

    data_manager = data_loader.OpsDataManager(
        experiments=experiment_dict,
        batch_size=1,
        data_split=(1, 0, 0),
        out_channels=["mCherry"],
        initial_yx_patch_size=(256, 256),
        verbose=verbose,
    )

    data_manager.construct_dataloaders(
        num_workers=num_workers, dataset_type="cell_profile"
    )
    train_loader = data_manager.train_loader
    batch = next(iter(train_loader))
    print("made a batch")

    measurements = get_core_measurements()
    shape_fcn = measurements.pop("sizeshape", None)

    def _process_batch(batch):

        img = batch["data"][0].detach().cpu().numpy()
        mask = batch["mask"][0].detach().cpu().numpy()
        features = single_channel_features(
            img, mask, prefix="mCherry", measurements=measurements
        )
        cell_mask_features = shape_features(
            img, mask, prefix="cell_mask", fcn=shape_fcn
        )
        features.update(cell_mask_features)
        features.update({"label_int": batch["gene_label"][0].detach().cpu().numpy()})
        features.update(
            {
                "label_str": data_manager.int_label_lut[
                    int(batch["gene_label"][0].detach().cpu().numpy())
                ]
            }
        )

        return pd.DataFrame(features)

    all_features = Parallel(n_jobs=1)(
        delayed(_process_batch)(batch) for batch in tqdm(train_loader)
    )

    all_features_df = pd.concat(all_features, ignore_index=True)
    if verbose:
        print(f"{len(all_features_df)} cells measured")

    # TODO: need to handle nans in a smart way
    all_features_df = all_features_df.fillna(0)

    # save_cp_features(experiment_dict, all_features_df, override=True)

    return all_features_df


def trim_features():
    """
    Some CP-measure features dont make sense to use in the context of our analysis,
    these must be removed before downstream tasks
    """

    # remove BoundingBoxMinimum/Maximum
    # Orientation

    return


if __name__ == "__main__":
    experiment_dict = {"ops0033_20250429": ["A/1/0", "A/2/0", "A/3/0"]}
    cp_features(experiment_dict=experiment_dict, verbose=True, num_workers=1)

# %%
