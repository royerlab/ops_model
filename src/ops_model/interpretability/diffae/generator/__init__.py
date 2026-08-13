"""DiffAE (diffusion autoencoder) — the DiffEx generator stage.

Semantic encoder + conditional UNet decoder, trained jointly on broad phase crops.
See README.md. The next stage (contrastive direction discovery) builds on the
trained z_sem latent + the classifier score.
"""
