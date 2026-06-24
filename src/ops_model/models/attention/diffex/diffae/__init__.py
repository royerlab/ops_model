"""DiffAE (diffusion autoencoder) — the DiffEx generator stage.

Semantic encoder + conditional UNet decoder, trained jointly on broad phase crops.
See ../PLAN.md §2 and README.md. The next stage (contrastive direction discovery,
§3) builds on the trained z_sem latent + the classifier score.
"""
