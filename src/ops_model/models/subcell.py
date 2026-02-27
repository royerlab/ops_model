"""SubCell bg (DNA + Protein) ViT-MAE model wrapper for OPS feature extraction.

This module provides a self-contained inference wrapper for the SubCell foundation
model (Human Protein Atlas, Blue+Green / DNA+Protein variant). Architecture code
is adapted from SubCellPortable (https://github.com/CellProfiling/SubCellPortable).

Weights are downloaded on first use from the public S3 bucket:
    s3://czi-subcell-public/models/DNA-Protein_MAE-CellS-ProtS-Pool.pth

Local cache:
    /hpc/projects/icd.ops/models/model_checkpoints/subcell/bg/

Channel mapping:
    batch["data"] channel 0  →  Blue  (DNA / DAPI)
    batch["data"] channel 1  →  Green (Protein-of-interest)

Output:
    (B, 1536) float32 embeddings from the Gated Attention Pooler
    (2 attention heads × 768 hidden dims = 1536)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from transformers.modeling_outputs import BaseModelOutput
from transformers.models.vit.configuration_vit import ViTConfig
from transformers.models.vit.modeling_vit import (
    BaseModelOutputWithPooling,
    ViTAttention,
    ViTEmbeddings,
    ViTIntermediate,
    ViTOutput,
    ViTPatchEmbeddings,
    ViTPooler,
    ViTPreTrainedModel,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Architecture (adapted from SubCellPortable/vit_model.py)
# Source: https://github.com/CellProfiling/SubCellPortable
# ---------------------------------------------------------------------------


@dataclass
class ViTPoolModelOutput:
    attentions: Tuple[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    pool_op: torch.FloatTensor = None
    pool_attn: torch.FloatTensor = None
    probabilities: torch.FloatTensor = None


class GatedAttentionPooler(nn.Module):
    def __init__(
        self, dim: int, int_dim: int = 512, num_heads: int = 1, out_dim: int = None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.attention_v = nn.Sequential(nn.Linear(dim, int_dim), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(dim, int_dim), nn.GELU())
        self.attention = nn.Linear(int_dim, num_heads)
        self.softmax = nn.Softmax(dim=-1)

        if out_dim is None:
            self.out_dim = dim * num_heads
            self.out_proj = nn.Identity()
        else:
            self.out_dim = out_dim
            self.out_proj = nn.Linear(dim * num_heads, out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[Tensor, Tensor]:
        v = self.attention_v(x)
        u = self.attention_u(x)
        attn = self.attention(v * u).permute(0, 2, 1)
        attn = self.softmax(attn)
        x = torch.bmm(attn, x)
        x = x.view(x.shape[0], -1)
        x = self.out_proj(x)
        return x, attn


class ViTLayer(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViTAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.layernorm_after = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        hidden_states = attention_output + hidden_states
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states)
        outputs = (layer_output,) + outputs
        return outputs


class ViTEncoder(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [ViTLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(
                hidden_states, layer_head_mask, output_attentions
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class ViTInferenceModel(ViTPreTrainedModel):
    def __init__(
        self,
        config: ViTConfig,
        add_pooling_layer: bool = True,
        use_mask_token: bool = False,
    ):
        super().__init__(config)
        self.config = config
        self.embeddings = ViTEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = ViTEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self) -> ViTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            head_outputs = (
                (sequence_output, pooled_output)
                if pooled_output is not None
                else (sequence_output,)
            )
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class ViTPoolClassifier(nn.Module):
    def __init__(self, config: Dict):
        super(ViTPoolClassifier, self).__init__()

        vit_model_config = config["vit_model"].copy()
        vit_model_config["attn_implementation"] = "eager"
        self.vit_config = ViTConfig(**vit_model_config)

        self.encoder = ViTInferenceModel(self.vit_config, add_pooling_layer=False)

        pool_config = config.get("pool_model")
        self.pool_model = GatedAttentionPooler(**pool_config) if pool_config else None

        self.out_dim = (
            self.pool_model.out_dim if self.pool_model else self.vit_config.hidden_size
        )
        self.num_classes = config["num_classes"]
        self.sigmoid = nn.Sigmoid()
        self.classifiers = nn.ModuleList([])

    def make_classifier(self):
        return nn.Sequential(
            nn.Linear(self.out_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.num_classes),
        )

    def load_model_dict(
        self,
        encoder_path: str,
        classifier_paths: Union[str, List[str]],
        device: str = "cpu",
    ):
        checkpoint = torch.load(encoder_path, map_location=device, weights_only=False)

        encoder_ckpt = {
            k[len("encoder.") :]: v for k, v in checkpoint.items() if "encoder." in k
        }
        status = self.encoder.load_state_dict(encoder_ckpt)
        logger.info(f"Encoder status: {status}")

        pool_ckpt = {
            k.replace("pool_model.", ""): v
            for k, v in checkpoint.items()
            if "pool_model." in k
        }
        pool_ckpt = {k.replace("1.", "0."): v for k, v in pool_ckpt.items()}
        if pool_ckpt and self.pool_model:
            status = self.pool_model.load_state_dict(pool_ckpt)
            logger.info(f"Pool model status: {status}")
        else:
            logger.info("No pool model found in checkpoint")

        if isinstance(classifier_paths, str):
            classifier_paths = [classifier_paths]

        self.classifiers = nn.ModuleList(
            [self.make_classifier() for _ in range(len(classifier_paths))]
        )
        for i, classifier_path in enumerate(classifier_paths):
            classifier_ckpt = torch.load(
                classifier_path, map_location=device, weights_only=False
            )
            classifier_ckpt = {
                k.replace("3.", "2."): v for k, v in classifier_ckpt.items()
            }
            classifier_ckpt = {
                k.replace("6.", "4."): v for k, v in classifier_ckpt.items()
            }
            status = self.classifiers[i].load_state_dict(classifier_ckpt)
            logger.info(f"Classifier {i + 1} status: {status}")

    def forward(self, x: torch.Tensor) -> ViTPoolModelOutput:
        b, c, h, w = x.shape
        outputs = self.encoder(x, output_attentions=True, interpolate_pos_encoding=True)

        if self.pool_model:
            pool_op, pool_attn = self.pool_model(outputs.last_hidden_state)
        else:
            pool_op = torch.mean(outputs.last_hidden_state, dim=1)
            pool_attn = None

        if len(self.classifiers) > 0:
            probs = torch.stack(
                [self.sigmoid(classifier(pool_op)) for classifier in self.classifiers],
                dim=1,
            )
            probs = torch.mean(probs, dim=1)
        else:
            probs = torch.zeros(
                pool_op.shape[0], self.num_classes, device=pool_op.device
            )

        h_feat = h // self.vit_config.patch_size
        w_feat = w // self.vit_config.patch_size

        attentions = outputs.attentions[-1][:, :, 0, 1:].reshape(
            b, self.vit_config.num_attention_heads, h_feat, w_feat
        )

        if pool_attn is not None:
            pool_attn = pool_attn[:, :, 1:].reshape(
                b, self.pool_model.num_heads, h_feat, w_feat
            )

        return ViTPoolModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            attentions=attentions,
            pool_op=pool_op,
            pool_attn=pool_attn,
            probabilities=probs,
        )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = Path("/hpc/projects/icd.ops/models/model_checkpoints/subcell/bg")
ENCODER_FILENAME = "DNA-Protein_MAE-CellS-ProtS-Pool.pth"
ENCODER_S3_KEY = "models/DNA-Protein_MAE-CellS-ProtS-Pool.pth"
S3_BUCKET = "czi-subcell-public"

# Model config sourced from:
#   SubCellPortable/models/bg/mae_contrast_supcon_model/model_config.yaml
BG_MAE_MODEL_CONFIG: Dict = {
    "vit_model": {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "image_size": 448,
        "patch_size": 16,
        "num_channels": 2,
        "qkv_bias": True,
    },
    "pool_model": {
        "dim": 768,
        "int_dim": 512,
        "num_heads": 2,
    },
    "num_classes": 31,
}


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class SubCellModel:
    """Inference wrapper for the SubCell bg (DNA + Protein) MAE-SupCon model.

    Produces 1536-dimensional single-cell embeddings from two-channel fluorescence
    images (DAPI → Blue channel 0, protein-of-interest → Green channel 1).

    On first use, encoder weights (~300 MB) are downloaded from:
        s3://czi-subcell-public/models/DNA-Protein_MAE-CellS-ProtS-Pool.pth
    and cached at CHECKPOINT_DIR.

    Input:
        batch["data"]: (B, 2, H, W) — channel 0 = DAPI, channel 1 = protein
    Output:
        (B, 1536) float32 tensor on CPU

    Architecture:
        ViT-B/16 encoder (hidden_size=768, 12 layers, patch_size=16)
        → Gated Attention Pooler (2 heads × 768 dims = 1536-dim output)
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = self._load_model()
        self.model.eval().to(device)

    def _download_weights(self) -> Path:
        """Download encoder weights from public S3 if not present locally."""
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        encoder_path = CHECKPOINT_DIR / ENCODER_FILENAME

        if encoder_path.exists():
            return encoder_path

        try:
            import boto3
            from botocore import UNSIGNED
            from botocore.config import Config as BotocoreConfig
        except ImportError as e:
            raise ImportError(
                "boto3 is required to download SubCell weights. "
                "Install with: pip install boto3"
            ) from e

        print(f"Downloading SubCell encoder weights to {encoder_path} ...")
        s3 = boto3.client("s3", config=BotocoreConfig(signature_version=UNSIGNED))
        s3.download_file(S3_BUCKET, ENCODER_S3_KEY, str(encoder_path))
        print("Download complete.")
        return encoder_path

    def _load_model(self) -> ViTPoolClassifier:
        encoder_path = self._download_weights()
        model = ViTPoolClassifier(BG_MAE_MODEL_CONFIG)
        model.load_model_dict(
            encoder_path=str(encoder_path),
            classifier_paths=[],
            device="cpu",
        )
        return model

    @staticmethod
    def preprocess(x: torch.Tensor) -> torch.Tensor:
        """Resize to 448×448 and apply per-channel min-max normalization.

        Declared as a staticmethod so callers that only need preprocessing
        (e.g. visualisation scripts) can call SubCellModel.preprocess(x)
        without loading model weights.

        Args:
            x: (B, C, H, W) float tensor.

        Returns:
            (B, C, 448, 448) float tensor, each channel normalized to [0, 1].
        """
        x = x.float()
        x = F.interpolate(x, size=(448, 448), mode="bilinear", align_corners=False)
        mn = x.flatten(2).min(dim=2).values[..., None, None]
        mx = x.flatten(2).max(dim=2).values[..., None, None]
        return (x - mn) / (mx - mn + 1e-8)

    def extract_features(self, batch: dict) -> torch.Tensor:
        """Extract 1536-dim SubCell embeddings from a data batch.

        Args:
            batch: dict with key "data" containing (B, 2, H, W) tensor,
                   channels ordered as [DAPI, protein-of-interest].

        Returns:
            (B, 1536) float32 tensor on CPU.
        """
        x = self.preprocess(batch["data"].to(self.device))
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                output = self.model(x)
        return output.pool_op.cpu()
