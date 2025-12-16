import inspect
from collections.abc import Collection
from copy import copy
from pathlib import Path
from typing import Optional, Sequence, Union

import lightning as L
import numpy as np
import torch
import zarr
import pandas as pd
from cytoself.components.blocks.fc_block import FCblock
from cytoself.components.layers.vq import VectorQuantizer
from cytoself.trainer.autoencoder.decoders.resnet2d import DecoderResnet
from cytoself.trainer.autoencoder.encoders.efficientenc2d import efficientenc_b0
from lightning.pytorch.callbacks import BasePredictionWriter

from torch import Tensor, nn, optim
import torchvision.utils as tvutils
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize


"""
Cytoself by default uses a split EfficientNet B0 model as two encoders.
Therefore, it needs two sets of block arguments for generating two EfficientNets.
"""


class CytoselfPredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: str,
        write_interval,
        int_label_lut: dict,
    ):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)
        self.high_count = 0
        self.low_count = 0
        self.metadata = {}
        self.int_label_lut = int_label_lut

    def setup(self, trainer, pl_module, stage):
        if stage == "predict":

            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.emb_local_dir = self.output_dir / "emb_2_chunks"
            self.emb_local_dir.mkdir(parents=True, exist_ok=True)

            self.classificatioin_score_dir = self.output_dir / "classification_scores"
            self.classificatioin_score_dir.mkdir(parents=True, exist_ok=True)

            self.global_emb_store = zarr.open_group(
                self.output_dir / f"global_emb.zarr", mode="w"
            )
            self.global_emb_store.create_group(self.high_count)

            self.emb_global_meta_dir = self.output_dir / "global_emb_metadata"
            self.emb_global_meta_dir.mkdir(parents=True, exist_ok=True)

            self.recon_store = zarr.open_group(
                self.output_dir / f"reconstructions.zarr", mode="w"
            )
        return

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,  # index of the sample within the epoch
        batch,
        batch_idx,
        dataloader_idx,
    ):

        recon, scores, emb_1, emb_2 = prediction
        if batch_idx == 0:
            self.recon_store.create_dataset(
                "reconstruction",
                data=recon.detach().cpu().numpy(),
                shape=recon.shape,
                chunks=(1,) + recon.shape[1:],
            )
            self.recon_store.create_dataset(
                "original",
                data=batch["data"].detach().cpu().numpy(),
                shape=batch["data"].shape,
                chunks=(1,) + batch["data"].shape[1:],
            )

        self.global_emb_store[self.high_count].create_dataset(
            self.low_count,
            data=emb_1.detach().cpu().numpy(),
            shape=emb_1.shape,
            chunks=(1,) + emb_1.shape[1:],
        )

        metadata_df = pd.DataFrame(
            {
                "label_int": batch["gene_label"].cpu().numpy(),
                "label_str": [
                    self.int_label_lut[label]
                    for label in batch["gene_label"].cpu().numpy()
                ],
                "sgRNA": [a["sgRNA"] for a in batch["crop_info"]],
                "experiment": [a["store_key"] for a in batch["crop_info"]],
                "x_position": [a["x_pheno"] for a in batch["crop_info"]],
                "y_position": [a["y_pheno"] for a in batch["crop_info"]],
                "well": [a["well"] + "_" + a["store_key"] for a in batch["crop_info"]],
            }
        )

        # save global embedding csvs
        emb_2_df = pd.DataFrame(
            emb_2.detach().cpu().numpy().reshape(emb_2.shape[0], -1)
        )
        emb_2_df = pd.concat([emb_2_df, metadata_df], axis=1)
        emb_2_csv_path = (
            self.emb_local_dir
            / f"global_emb_batch_{self.high_count}_{self.low_count}.csv"
        )
        emb_2_df.to_csv(emb_2_csv_path, index=False)

        # save classification score csvs
        scores_df = pd.DataFrame(scores.detach().cpu().numpy())
        scores_df = pd.concat([scores_df, metadata_df], axis=1)
        scores_csv_path = (
            self.classificatioin_score_dir
            / f"classification_scores_batch_{self.high_count}_{self.low_count}.csv"
        )
        scores_df.to_csv(scores_csv_path, index=False)

        # save global_emb metadata
        global_emb_df = pd.DataFrame(
            {
                "batch_index": batch_indices,
                "low_count": self.low_count,
                "high_count": self.high_count,
            }
        )
        global_emb_df = pd.concat([global_emb_df, metadata_df], axis=1)
        global_emb_csv_path = (
            self.emb_global_meta_dir
            / f"global_emb_metadata_batch_{self.high_count}_{self.low_count}.csv"
        )
        global_emb_df.to_csv(global_emb_csv_path, index=False)

        self.low_count += 1
        if self.low_count % 10 == 0:
            self.high_count += 1
            self.global_emb_store.create_group(self.high_count)
            self.low_count = 0

        return


class LitCytoSelf(L.LightningModule):

    def __init__(
        self,
        # TODO: change these hardcodings!!
        emb_shapes: Collection[tuple[int, int]] = ((32, 32), (4, 4)),
        vq_args: Union[dict, Collection[dict]] = {
            "num_embeddings": 512,
            "embedding_dim": 64,
        },
        num_class: int = 1000,
        input_shape: Optional[tuple[int, int, int]] = (1, 128, 128),
        output_shape: Optional[tuple[int, int, int]] = (1, 128, 128),
        fc_input_type: str = "vqvec",
        fc_output_idx: Union[str, Sequence[int]] = [2],
        vq_coeff: float = 1.0,
        fc_coeff: float = 1.0,
    ):
        super().__init__()
        self.model = CytoselfFull(
            emb_shapes=emb_shapes,
            vq_args=vq_args,
            num_class=num_class,
            input_shape=input_shape,
            output_shape=output_shape,
            fc_input_type=fc_input_type,
            fc_output_idx=fc_output_idx,
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.variance = (
            1  # Should be 1 because we normalize the data to have 0 mean and std of 1
        )
        self.vq_coeff = vq_coeff
        self.fc_coeff = fc_coeff

    def forward(self, batch):
        img = batch["data"]
        return self.model(img)

    def training_step(self, batch):
        img = batch["data"]
        lab = batch["gene_label"]

        model_outputs = self.model(img)
        loss, mse_loss, vq_loss, fc_loss = self._calc_losses(
            model_outputs=model_outputs,
            img=img,
            lab=lab,
            variance=self.variance,
            vq_coeff=self.vq_coeff,
            fc_coeff=self.fc_coeff,
        )
        self.log_dict(
            {
                "train/total_loss": loss,
                "train/mse_loss": mse_loss,
                "train/vq_loss": vq_loss,
                "train/fc_loss": fc_loss,
                "train/extra/perplexity1": self.model.perplexity["perplexity1"].item(),
                "train/extra/perplexity2": self.model.perplexity["perplexity2"].item(),
                "train/extra/vq1_commitment_loss": self.model.vq_loss["vq1"][
                    "commitment_loss"
                ].item(),
                "train/extra/vq2_commitment_loss": self.model.vq_loss["vq2"][
                    "commitment_loss"
                ].item(),
                "train_extra/vq1_quantization_loss": self.model.vq_loss["vq1"][
                    "quantization_loss"
                ].item(),
                "train_extra/vq2_quantization_loss": self.model.vq_loss["vq2"][
                    "quantization_loss"
                ].item(),
                "train/extra/vq1_softmax_loss": self.model.vq_loss["vq1"][
                    "softmax_loss"
                ].item(),
                "train/extra/vq2_softmax_loss": self.model.vq_loss["vq2"][
                    "softmax_loss"
                ].item(),
            },
            batch_size=img.size(0),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        img = batch["data"]
        lab = batch["gene_label"]

        model_outputs = self.model(img)
        loss, mse_loss, vq_loss, fc_loss = self._calc_losses(
            model_outputs,
            img,
            lab,
            variance=self.variance,
            vq_coeff=self.vq_coeff,
            fc_coeff=self.fc_coeff,
        )
        self.log_dict(
            {
                "val/total_loss": loss,
                "val/mse_loss": mse_loss,
                "val/vq_loss": vq_loss,
                "val/fc_loss": fc_loss,
            },
            batch_size=img.size(0),
        )

        if batch_idx == 0:
            self.log_images(img, model_outputs[0])

        return loss

    def predict_step(self, batch):
        data = batch["data"].cuda()
        recon, prediction = self.model(data)
        emb_1 = self.model(data, output_layer="vqvec1")
        emb_2 = self.model(data, output_layer="vqvec2")
        return recon, prediction, emb_1, emb_2

    def configure_optimizers(self):
        return self.optimizer

    def _calc_losses(
        self,
        model_outputs: tuple,
        img: Tensor,
        lab: Tensor,
        vq_coeff: float,
        fc_coeff: float,
        **kwargs: dict,
    ):
        mse_loss_fn = nn.MSELoss()
        ce_loss_fn = nn.CrossEntropyLoss()

        # reconstruction loss comparing output images to input images
        self.model.mse_loss["reconstruction1_loss"] = (
            mse_loss_fn(model_outputs[0], img) / self.variance
        )

        # classification loss for each class
        self.model.fc_loss = {
            f"fc{self.model.fc_output_idx[j]}_loss": ce_loss_fn(t, i)
            for j, (t, i) in enumerate(
                zip(model_outputs[1:], [lab] * (len(model_outputs) - 1))
            )
        }
        return self._combine_losses(vq_coeff, fc_coeff)

    def _combine_losses(self, vq_coeff: float, fc_coeff: float):
        fc_loss_list = list(self.model.fc_loss.values())
        vq_loss_list = [d["loss"] for d in self.model.vq_loss.values()]
        mse_loss_list = list(self.model.mse_loss.values())
        mse_loss = torch.stack(mse_loss_list).sum() if len(mse_loss_list) > 0 else 0
        vq_loss = torch.stack(vq_loss_list).sum() if len(vq_loss_list) > 0 else 0
        fc_loss = torch.stack(fc_loss_list).sum() if len(fc_loss_list) > 0 else 0
        loss = +mse_loss + vq_coeff * vq_loss + fc_coeff * fc_loss
        return loss, mse_loss, vq_loss, fc_loss

    def log_images(self, inputs, outputs, num_samples=8):
        n = min(inputs.size(0), num_samples)
        grid = tvutils.make_grid(
            torch.cat((inputs[:n], outputs[:n])),
            nrow=n,
            normalize=True,
            value_range=(-1, 1),
        )
        self.logger.experiment.add_image("reconstructions", grid, self.current_epoch)


class CytoselfFull(nn.Module):
    """
    Cytoself original model (2-stage encoder & decoder with 2 VQ layers and 2 fc blocks)
    EfficientNet_B0 is used for encoders for the sake of saving computation resource.
    """

    def __init__(
        self,
        emb_shapes: Collection[tuple[int, int]],
        vq_args: Union[dict, Collection[dict]],
        num_class: int,
        input_shape: Optional[tuple[int, int, int]] = None,
        output_shape: Optional[tuple[int, int, int]] = None,
        fc_input_type: str = "vqvec",
        fc_output_idx: Union[str, Sequence[int]] = "all",
        encoder_args: Optional[Collection[dict]] = None,
        decoder_args: Optional[Collection[dict]] = None,
        fc_args: Optional[Union[dict, Collection[dict]]] = None,
        encoders: Optional[Collection] = None,
        decoders: Optional[Collection] = None,
    ) -> None:
        """
        Construct a cytoself full model

        Parameters
        ----------
        emb_shapes : tuple or list of tuples
            Embedding tensor shape except for channel dim
        vq_args : dict
            Additional arguments for the Vector Quantization layer
        num_class : int
            Number of output classes for fc layers
        input_shape : tuple of int
            Input tensor shape
        output_shape : tuple of int
            Output tensor shape; will be same as input_shape if None.
        fc_input_type : str
            Input type for the fc layers;
            vqvec: quantized vector, vqind: quantized index, vqindhist: quantized index histogram
        fc_output_idx : int or 'all'
            Index of encoder to connect FC layers
        encoder_args : dict
            Additional arguments for encoder
        decoder_args : dict
            Additional arguments for decoder
        fc_args : dict
            Additional arguments for fc layers
        encoders : encoder module
            (Optional) Custom encoder module
        decoders : decoder module
            (Optional) Custom decoder module
        """
        super().__init__()
        # Check vq_args and emb_shapes and compute emb_ch_splits
        vq_args = duplicate_kwargs(vq_args, emb_shapes)
        vq_args, emb_shapes = calc_emb_dim(vq_args, emb_shapes)
        self.emb_shapes = emb_shapes

        # Construct encoders (from the one close to input data to the one to far from)
        if encoders is None:
            self.encoders = self._const_encoders(input_shape, encoder_args)
        else:
            self.encoders = encoders

        # Construct decoders (shallow to deep)
        if decoders is None:
            if output_shape is None:
                output_shape = input_shape
            self.decoders = self._const_decoders(output_shape, decoder_args)
        else:
            self.decoders = decoders

        # Construct VQ layers (same order as encoders)
        self.vq_layers = nn.ModuleList()
        for i, varg in enumerate(vq_args):
            self.vq_layers.append(VectorQuantizer(**varg))
        self.vq_loss = None
        self.perplexity = None
        self.mse_loss = None

        # Construct fc blocks (same order as encoders)
        fc_args_default = {"num_layers": 1, "num_features": 1000}
        if fc_args is None:
            fc_args = {}
        fc_args.update({k: v for k, v in fc_args_default.items() if k not in fc_args})
        fc_args = duplicate_kwargs(fc_args, emb_shapes)

        self.fc_layers = nn.ModuleList()
        for i, shp in enumerate(emb_shapes):
            arg = fc_args[i]
            if fc_input_type == "vqind":
                arg["in_channels"] = np.prod(shp[1:])
            elif fc_input_type == "vqindhist":
                arg["in_channels"] = vq_args[i]["num_embeddings"]
            else:
                arg["in_channels"] = np.prod(shp)
            arg["out_channels"] = num_class
            self.fc_layers.append(FCblock(**arg))
        self.fc_loss = None
        self.fc_input_type = fc_input_type
        if fc_output_idx == "all":
            self.fc_output_idx = [i + 1 for i in range(len(emb_shapes))]
        else:
            self.fc_output_idx = fc_output_idx

    def _const_encoders(self, input_shape, encoder_args) -> nn.ModuleList:
        """
        Constructs a Module list of encoders

        Parameters
        ----------
        input_shape : tuple
            Input tensor shape
        encoder_args : dict
            Additional arguments for encoder

        Returns
        -------
        nn.ModuleList

        """
        if encoder_args is None:
            encoder_args = default_block_args
        length_checker(self.emb_shapes, encoder_args)

        encoders = nn.ModuleList()
        for i, shp in enumerate(self.emb_shapes):
            encoder_args[i].update(
                {
                    "in_channels": (
                        input_shape[0] if i == 0 else self.emb_shapes[i - 1][0]
                    ),
                    "out_channels": shp[0],
                    "first_layer_stride": 2 if i == 0 else 1,
                }
            )
            encoders.append(efficientenc_b0(**encoder_args[i]))
        return encoders

    def _const_decoders(self, output_shape, decoder_args) -> nn.ModuleList:
        """
        Constructs a Module list of decoders

        Parameters
        ----------
        output_shape : tuple
            Output tensor shape
        decoder_args : dict
            Additional arguments for decoder

        Returns
        -------
        nn.ModuleList

        """
        if decoder_args is None:
            decoder_args = [{}] * len(self.emb_shapes)

        decoders = nn.ModuleList()
        for i, shp in enumerate(self.emb_shapes):
            if i == 0:
                shp = (sum(i[0] for i in self.emb_shapes),) + shp[1:]
            decoder_args[i].update(
                {
                    "input_shape": shp,
                    "output_shape": output_shape if i == 0 else self.emb_shapes[i - 1],
                    "linear_output": i == 0,
                }
            )
            decoders.append(DecoderResnet(**decoder_args[i]))
        return decoders

    def _connect_decoders(self, encoded_list):
        decoding_list = []
        for i, (encd, dec) in enumerate(zip(encoded_list[::-1], self.decoders[::-1])):
            if i < len(self.decoders) - 1:
                decoding_list.append(
                    resize(
                        encd,
                        self.emb_shapes[0][1:],
                        interpolation=InterpolationMode.NEAREST,
                    )
                )
                self.mse_loss[f"reconstruction{len(self.decoders) - i}_loss"] = (
                    nn.MSELoss()(dec(encd), encoded_list[-2 - i])
                )
            else:
                decoding_list.append(encd)
                decoded_final = dec(torch.cat(decoding_list, 1))
        return decoded_final

    def forward(
        self, x: Tensor, output_layer: str = "decoder0"
    ) -> tuple[Tensor, Tensor]:
        """
        Cytoselflite model consists of encoders & decoders such as:
        encoder1 -> vq layer1 -> encoder2 -> vq layer2 -> ... -> decoder2 -> decoder1
        The order in the Module list of encoders and decoders is always 1 -> 2 -> ...

        Parameters
        ----------
        x : Tensor
            Image data
        output_layer : str
            Name of layer + index (integer) as the exit layer.
            This is used to indicate the exit layer for output embeddings.

        Returns
        -------
        A list of Tensors

        """
        self.vq_loss = {}
        self.perplexity = {}
        self.mse_loss = {}
        out_layer_name, out_layer_idx = output_layer[:-1], int(output_layer[-1]) - 1

        fc_outs = []
        encoded_list = []
        for i, enc in enumerate(self.encoders):
            encoded = enc(x)
            if out_layer_name == "encoder" and i == out_layer_idx:
                return encoded

            (
                vq_loss,
                quantized,
                perplexity,
                _,
                _encoding_indices,
                _index_histogram,
                softmax_histogram,
            ) = self.vq_layers[i](encoded)
            if i == out_layer_idx:
                if out_layer_name == "vqvec":
                    return quantized
                elif out_layer_name == "vqind":
                    return _encoding_indices
                elif out_layer_name == "vqindhist":
                    return _index_histogram

            if i + 1 in self.fc_output_idx:
                if self.fc_input_type == "vqvec":
                    fcout = self.fc_layers[i](quantized.reshape(quantized.size(0), -1))
                elif self.fc_input_type == "vqind":
                    fcout = self.fc_layers[i](
                        _encoding_indices.reshape(_encoding_indices.size(0), -1)
                    )
                elif self.fc_input_type == "vqindhist":
                    fcout = self.fc_layers[i](softmax_histogram)
                else:
                    fcout = self.fc_layers[i](encoded.reshape(encoded.size(0), -1))
                fc_outs.append(fcout)
            encoded_list.append(quantized)
            self.vq_loss[f"vq{i + 1}"] = vq_loss
            self.perplexity[f"perplexity{i + 1}"] = perplexity
            x = encoded

        decoded_final = self._connect_decoders(encoded_list)
        return tuple([decoded_final] + fc_outs)


default_block_args = [
    # block arguments for the first encoder
    {
        "blocks_args": [
            {
                "expand_ratio": 1,
                "kernel": 3,
                "stride": 1,
                "input_channels": 32,
                "out_channels": 16,
                "num_layers": 1,
            },
            {
                "expand_ratio": 6,
                "kernel": 3,
                "stride": 2,
                "input_channels": 16,
                "out_channels": 24,
                "num_layers": 2,
            },
            {
                "expand_ratio": 6,
                "kernel": 5,
                "stride": 1,
                "input_channels": 24,
                "out_channels": 40,
                "num_layers": 2,
            },
        ]
    },
    # block arguments for the second encoder
    {
        "blocks_args": [
            {
                "expand_ratio": 6,
                "kernel": 3,
                "stride": 2,
                "input_channels": 40,
                "out_channels": 80,
                "num_layers": 3,
            },
            {
                "expand_ratio": 6,
                "kernel": 5,
                "stride": 2,  # 1 in the original
                "input_channels": 80,
                "out_channels": 112,
                "num_layers": 3,
            },
            {
                "expand_ratio": 6,
                "kernel": 5,
                "stride": 2,
                "input_channels": 112,
                "out_channels": 192,
                "num_layers": 4,
            },
            {
                "expand_ratio": 6,
                "kernel": 3,
                "stride": 1,
                "input_channels": 192,
                "out_channels": 320,
                "num_layers": 1,
            },
        ]
    },
]


def length_checker(collection1: Collection, collection2: Collection):
    """
    Checks if two variables have the same length.
    Specifically to check if the number of parameters match with the number of encoders.

    Parameters
    ----------
    collection1 : Collection of arguments
        Encoder dependent argument whose number should match with the number of encoders
    collection2 : Collection of arguments
        Encoder dependent argument whose number should match with the number of encoders

    Returns
    -------
    none

    """
    if len(collection1) != len(collection2):
        raise ValueError(
            f"{collection1=}".split("=")[0]
            + " and "
            + f"{collection2=}".split("=")[0]
            + " must be the same length."
        )


def duplicate_kwargs(arg1: Collection, arg2: Collection):
    """
    Duplicate arg1 by the length of arg2.

    Parameters
    ----------
    arg1 : Collection of arguments
        Encoder dependent argument whose number should match with the number of encoders
    arg2 : Collection of arguments
        Encoder dependent argument whose number should match with the number of encoders

    Returns
    -------
    list

    """
    if isinstance(arg1, dict):
        arg1 = [arg1] * len(arg2)
    else:
        if len(arg1) != len(arg2):
            raise ValueError(
                f"The length of arg1 {len(arg1)} must match with that of arg2 {len(arg2)}."
            )
    return arg1


def calc_emb_dim(vq_args: Sequence[dict], emb_shapes: Collection[tuple[int, int]]):
    """
    Calculate embedding dimensions

    Parameters
    ----------
    vq_args : Sequence of dict
        A list of arguments for VQ layers in dict format
    emb_shapes : tuple or list of tuples
        Embedding tensor shape except for channel dim

    Returns
    -------
    vq_args : Sequence of dict where channel_split is added if it wasn't there
    emb_shapes_out : embedding dimensions were added to emb_shapes

    """
    vq_args_out = copy(vq_args)
    emb_shapes_out = []
    for i, varg in enumerate(vq_args_out):
        if "embedding_dim" not in varg:
            raise ValueError("embedding_dim is required for vq_args.")
        if "channel_split" not in varg:
            varg["channel_split"] = (
                inspect.signature(VectorQuantizer).parameters["channel_split"].default
            )
        emb_shapes_out.append(
            (varg["embedding_dim"] * varg["channel_split"],) + tuple(emb_shapes[i])
        )
    return vq_args_out, tuple(emb_shapes_out)
