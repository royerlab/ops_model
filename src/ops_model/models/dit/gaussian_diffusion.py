import torch
import numpy as np

from ops_model.models.dit.diffusion_utils import (
    _extract_into_tensor,
    normal_kl,
    mean_flat,
    discretized_gaussian_log_likelihood,
)


def get_beta_schedule(
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    num_diffusion_timesteps: int = 1000,
):
    """
    DiT paper uses a linear beta schedule.
    Default args are set according to the DiT paper:
        beta_start=1e-4, beta_end=2e-2, num_diffusion_timesteps=1000
    """
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

    return betas


class GaussianDiffusion:
    def __init__(
        self,
        betas,
    ):

        self.betas = betas
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = (
            np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
            if len(self.posterior_variance) > 1
            else np.array([])
        )

        # For use computing x_0 from x_t and eps:
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.num_timesteps = betas.shape[0]

    def q_sample(self, x_start, t, noise=None):
        """
        Sample from the forward diffusion process at a given timestep.

            - q(x_t | x_0)
            - x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * N(0,I)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            - q(x_{t-1} | x_t, x_0)
        """
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _predict_xstart_from_eps(self, x_t, t, eps):

        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def p_mean_variance(
        self,
        model_output,
        x_t,
        t,
        denoised_fn=None,
        clip_denoised=True,
    ):
        B, C = x_t.shape[:2]

        model_output, model_var_values = torch.split(model_output, C, dim=1)
        min_log_var = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        max_log_var = _extract_into_tensor(np.log(self.betas), t, x_t.shape)
        frac = (
            model_var_values + 1
        ) / 2  # <--- model_var_values usually have a tanh activation
        # as the last layer [-1, 1], this maps it to [0, 1]
        model_log_variance = (
            frac * max_log_var + (1 - frac) * min_log_var
        )  # linear interpolation in log space
        model_variance = torch.exp(model_log_variance)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        pred_xstart = process_xstart(
            self._predict_xstart_from_eps(x_t=x_t, t=t, eps=model_output)
        )

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x_t, t=t
        )

        return model_mean, model_variance, model_log_variance, pred_xstart

    def variational_lb(
        self,
        model,
        x_start,
        x_t,
        t,
        y,
        clip_denoised=True,
        model_kwargs=None,
    ):

        model_output = model(x_t, t, y, **model_kwargs)

        # KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start, x_t, t
        )
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            model_output, x_t, t, clip_denoised=clip_denoised
        )

        kl = normal_kl(
            true_mean, true_log_variance_clipped, model_mean, model_log_variance
        )
        kl = mean_flat(kl) / np.log(2.0)  # convert to bits

        # -log p(x_0 | x_1)
        decoder_nll = -1 * discretized_gaussian_log_likelihood(
            x=x_start,
            means=model_mean,
            log_scales=0.5 * model_log_variance,
        )
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)

        return {"output": output, "pred_xstart": pred_xstart}

    def training_losses(
        self,
        model,
        x_start,
        t,
        y,
        noise=None,
        model_kwargs=None,
    ):
        # generate a noised image
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        loss_dict = {}

        model_output = model(x_t, t, y, **model_kwargs)
        B, C = x_t.shape[:2]
        model_output, model_var_values = torch.split(model_output, C, dim=1)
        frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
        loss_dict["vlb"] = self.variational_lb(
            model=lambda *args, **kwargs: frozen_out,
            x_start=x_start,
            x_t=x_t,
            t=t,
            y=y,
            model_kwargs=model_kwargs,
        )["output"]

        # target could also be x_start or x_t-1 depending on how the model is structured
        target = noise
        loss_dict["mse"] = mean_flat((target - model_output) ** 2)

        loss_dict["loss"] = loss_dict["mse"] + loss_dict["vlb"]

        return loss_dict
