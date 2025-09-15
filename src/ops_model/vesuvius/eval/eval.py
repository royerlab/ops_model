from collections import defaultdict
from pathlib import Path

import anndata as ad
import click
import lightning as L
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import yaml
import zarr
from lightning.pytorch import seed_everything
from ops_analysis.model.models import byol, cytoself_model, dynaclr
from ops_analysis.model.vesuvius import vesuvius_dataloader
from tqdm import tqdm

mp.set_sharing_strategy("file_system")


def load_data(embedding_dir, num_samples):

    embedding_dir = Path(embedding_dir)
    store = zarr.open(embedding_dir, mode="r")
    position_dict = store.attrs.asdict()
    position_keys = list(position_dict.keys())
    random_set = np.random.choice(position_keys, size=num_samples, replace=False)

    index_by_pos = defaultdict(list)
    entry_list = []
    for key in tqdm(random_set):
        entry = position_dict[key]
        index_by_pos[entry["position"]].append(entry["index"])
        entry_list.append(entry)

    entry_info = pd.DataFrame(entry_list)

    data_parts = []
    for pos, indices in tqdm(index_by_pos.items(), desc="Loading data"):
        try:
            group_array = store[pos]
        except:
            entry_info = entry_info[entry_info["position"] != pos]
            continue
        batch = group_array[indices]
        data_parts.append(batch)
    data = np.concatenate(data_parts, axis=0)
    data = data.reshape(data.shape[0], -1)
    print("Loaded data")

    return data, entry_info


def get_data_manager(config_path):

    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset_type = config["dataset_type"]

    data_manager = vesuvius_dataloader.VesuviusDataManager(
        batch_size=config["data_manager"]["batch_size"],
        data_split=(0.8, 0.09, 0.01),  # tuple(config["data_manager"]["data_split"]),
        in_channels=config["data_manager"]["in_channels"],
        final_yx_patch_size=tuple(config["data_manager"]["final_yx_patch_size"]),
    )
    data_manager.construct_dataloaders(
        num_workers=config["data_manager"]["num_workers"],
        dataset_kwargs=config["data_manager"].get("dataset_kwargs"),
    )

    return data_manager


def create_ann_data(embedding_path, num_samples, data_manager):

    data, entry_info = load_data(embedding_path, num_samples)

    adata = ad.AnnData(data)
    adata.obs["gene_name"] = [
        data_manager.labels_df.loc[a].gene_name for a in entry_info.total_index
    ]
    adata.obs["gene_int"] = [a for a in entry_info.gene_label]

    return adata


def reduce_dims(adata, num_components, save_dir):

    sc.settings.figdir = save_dir
    sc.tl.pca(adata, n_comps=num_components)
    sc.pl.pca_variance_ratio(adata, n_pcs=100, log=False, save=True)
    sc.pp.neighbors(adata, n_pcs=50, n_neighbors=15, metric="cosine")  # or euclidean
    sc.tl.umap(adata, min_dist=0.1)
    sc.pl.umap(adata, size=20, save=True)

    variance_ratio = adata.uns["pca"]["variance_ratio"]
    cumulative = np.cumsum(variance_ratio)
    n_90 = np.argmax(cumulative >= 0.9) + 1

    adata.write_zarr(save_dir / "adata.zarr")

    return n_90


def mean_similarity(
    adata,
):
    embeddings = torch.tensor(adata.X).cuda()
    x = F.normalize(embeddings, dim=1)
    sim_matrix = x @ x.T
    mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
    upper_triangle_values = sim_matrix[mask]
    mean_similarity = upper_triangle_values.mean().item()
    std_similarity = upper_triangle_values.std().item()

    return mean_similarity, std_similarity


def cov_matrix_rank(adata):
    # maybe normalize by embedding shape???
    embeddings = torch.tensor(adata.X).cuda()
    x = F.normalize(embeddings, dim=1)
    cov_matrix = torch.cov(x.T)
    rank = torch.linalg.matrix_rank(cov_matrix).item()

    sign, logdet = torch.slogdet(cov_matrix)
    if sign <= 0:
        logdet = float(
            "-inf"
        )  # Log determinant is not defined for non-positive definite matrices

    return rank, logdet


def mean_class_similarity(
    adata,
    save_dir,
    save=True,
):
    gene_labels = adata.obs["gene_name"].unique().tolist()
    similarities = {}
    for g in gene_labels:
        embeddings = torch.tensor(adata[adata.obs["gene_name"] == g].X).cuda()
        if embeddings.shape[0] < 2:
            continue

        x = F.normalize(embeddings, dim=1)
        sim_matrix = x @ x.T
        mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
        upper_triangle_values = sim_matrix[mask]
        mean_similarity = upper_triangle_values.mean()
        similarities[g] = mean_similarity.item()

    similarity_df = pd.DataFrame.from_dict(
        similarities, orient="index", columns=["mean_similarity"]
    ).sort_values(by="mean_similarity", ascending=False)

    if save:
        similarity_df.to_csv(save_dir / "cosine_similarity.csv")

    return similarity_df


def alignment_and_uniformity(adata):
    # Code adapted from:
    # title={Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere},
    # author={Wang, Tongzhou and Isola, Phillip},
    # booktitle={International Conference on Machine Learning},
    # organization={PMLR},
    # pages={9929--9939},
    # year={2020}

    gene_int_list = adata.obs["gene_int"].unique().tolist()
    x = []  # positive pair i
    y = []  # positive pair j
    for i in gene_int_list:
        single_gene_embs = adata.X[adata.obs["gene_int"] == i]
        x += [single_gene_embs[j] for j in range(len(single_gene_embs) - 1)]
        y += [
            single_gene_embs[z]
            for z in np.random.permutation(np.arange(len(single_gene_embs) - 1))
        ]

    x = torch.tensor(np.asarray(x))
    y = torch.tensor(np.asarray(y))
    alignment = (x - y).norm(p=2, dim=1).pow(2).mean()

    uniformity = torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    return alignment, uniformity


def cluster_analysis(adata):
    resolutions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2]
    n_clusters = []

    for r in resolutions:
        sc.tl.leiden(adata, resolution=r, key_added=f"leiden_{r}")
        n_clusters.append(adata.obs[f"leiden_{r}"].nunique())
        sc.pl.umap(adata, color=f"leiden_{r}", save=f"_leiden_{r}.png")

    return resolutions, n_clusters


def run_inference(
    config: dict,
    data_manager=None,
):

    model_type = config["model_type"]
    run_name = config["run_name"]

    test_loader = data_manager.test_loader

    torch.set_float32_matmul_precision("medium")  # huge boost in speed

    if model_type == "byol":
        lit_model = byol.LitBYOL.load_from_checkpoint(
            config["evaluation"]["model_checkpoint"]
        )
        pred_writer = byol.BYOLPredictionWriter

    if model_type == "cytoself":
        mc = config["model"]
        lit_model = cytoself_model.LitCytoSelf.load_from_checkpoint(
            "/hpc/projects/intracellular_dashboard/ops/models/model_checkpoints/cytoself/cytoself_20250627_1-2025-06-27-global_step=0.000000-val_loss=7.08-v2.ckpt",
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

    if model_type == "dynaclr":
        lit_model = dynaclr.LitDynaClr(
            lr=config["model"]["lr"],
            schedule=config["model"]["schedule"],
            log_batches_per_epoch=config["model"]["log_batches_per_epoch"],
            log_samples_per_batch=config["model"]["log_samples_per_batch"],
            log_embeddings=config["model"]["log_embeddings"],
            example_input_array_shape=tuple(
                config["model"]["example_input_array_shape"]
            ),
            **config["model"].get("encoder", {}),
        )
        pred_writer = dynaclr.DynaClrPredictionWriter

    trainer = L.Trainer(
        devices=1,
        accelerator="gpu",
        callbacks=[
            pred_writer(
                output_dir=f"/hpc/projects/intracellular_dashboard/ops/models/predictions/vesuvius/{model_type}",
                zarr_suffix=run_name,
                write_interval="batch",
            )
        ],
    )
    predictions = trainer.predict(lit_model, dataloaders=test_loader)

    return f"/hpc/projects/intracellular_dashboard/ops/models/predictions/vesuvius/{config['model_type']}/emb_{config['run_name']}.zarr"


def evaluate(
    config_path,
    num_samples: int = 1000,
    run_predict: bool = False,
    run_ann_data: bool = True,
):
    data_manager = get_data_manager(config_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    save_dir = Path(
        f"/hpc/projects/intracellular_dashboard/ops/models/predictions/vesuvius/{config['model_type']}/analyzed/{config['run_name']}"
    )

    if run_predict:
        print("Running Inference")
        embedding_path = run_inference(config, data_manager=data_manager)
    if run_ann_data or run_predict:
        print("Creating AnnData")
        adata = create_ann_data(
            embedding_path=f"/hpc/projects/intracellular_dashboard/ops/models/predictions/vesuvius/{config['model_type']}/emb_{config['run_name']}.zarr",
            num_samples=int(num_samples),
            data_manager=data_manager,
        )
    else:
        print("Loading AnnData")
        adata = ad.io.read_zarr(save_dir / "adata.zarr")
    print("Loaded Data")
    n_comps_90 = reduce_dims(
        adata,
        num_components=100,
        save_dir=save_dir,
    )

    similarity_df = mean_class_similarity(adata, save_dir=save_dir)
    mean_sim, std_sim = mean_similarity(adata)
    print("Calculated Mean Class Similarity")
    conv_rank, conv_det = cov_matrix_rank(adata)
    print("Calculated Covariance Matrix Rank and Determinant")
    alignment, uniformity = alignment_and_uniformity(adata)
    print("Calculated Alignment and Uniformity")
    resolutions, num_clusters = cluster_analysis(adata)
    print("Performed Cluster Analysis")

    results = {
        "similarity_df": similarity_df,
        "mean_similarity": (mean_sim, std_sim),
        "n_components_90": n_comps_90,
        "covariance": (conv_rank, conv_det),
        "alignment_uniformity": (alignment, uniformity),
        "cluster_analysis": (resolutions, num_clusters),
    }

    print(f"Number of components to explain 90% variance: {n_comps_90}")
    print(f"Mean similarity: {mean_sim:.3f}")
    print(f"Standard deviation of similarity: {std_sim:.3f}")
    print(f"Covariance Matrix Rank: {conv_rank}")
    print(f"Covariance Matrix Determinant: {conv_det}")
    print(f"Alignment: {alignment:.2f}")
    print(f"Uniformity: {uniformity:.3f}")
    print(f"Resolutions:        {resolutions}")
    print(f"Number of Clusters: {num_clusters}")

    # print("Attempting to train an mlp classifier off the embeddings")
    # eval_mlp.eval_mlp(
    #     save_dir / "adata.zarr", num_workers=config["data_manager"]["num_workers"]
    # )

    return results


@click.command()
@click.option("-c", "--config_path")
@click.option("-n", "--num_samples")
@click.option("-predict", "--run_predict", default=False)
@click.option("-ann", "--run_ann_data", default=True)
def evaluate_cli(config_path, num_samples, run_predict, run_ann_data):
    return evaluate(config_path, num_samples, run_predict, run_ann_data)


if __name__ == "__main__":
    seed_everything(43)
    evaluate_cli()
