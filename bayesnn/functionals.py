import math
import torch

def bayesian_kl_div(mu_1, sigma_1, mu_0, sigma_0):
    r"""
    Closed-form KL divergence KL(q || p) between two diagonal Gaussian distributions.

    Parameters
    ----------
    mu_1 :    torch.Tensor
              Posterior means.
    sigma_1 : torch.Tensor
              Posterior standard deviations.
    mu_0 :    torch.Tensor
              Prior mean(s) 
    sigma_0 : torch.Tensor
              Prior standard deviations
    """
    
    return 0.5 * ( 
        sigma_1.pow(2) / sigma_0.pow(2)
            + (mu_1 - mu_0).pow(2)/(sigma_0.pow(2))
                - 1 + 2 * torch.log(sigma_0/sigma_1)
    ).sum()

def _resolve_noise_variance(model, device, dtype):
    if hasattr(model, "noise_var"):
        return model.noise_var.to(device=device, dtype=dtype)
    if hasattr(model, "out_noise_var"):
        return torch.tensor(model.out_noise_var, device=device, dtype=dtype)
    raise AttributeError("Model must expose either `noise_var` buffer or `out_noise_var` scalar.")


def mc_loglikelihood_samples(model, x, y, mc=250):
    r"""
        Monte Carlo samples of the per-example Gaussian log-likelihood:
            log p(y | x, W, \Sigma) where w ~ q(W | D).

        Parameters
        ----------
        model : callable
            A stochastic model where model(x) returns the output 
            computed by a realization of the network weights
        x : torch.Tensor
            Input tensor (B, input_dim)
        y : torch.Tensor
            Observed outputs (B, D)
        mc : int
            Number of Monte Carlo samples
    
        Returns
        -------
        torch.Tensor
            Tensor with shape (mc, B) containing per-sample log-likelihoods.
    """

    device, dtype = x.device, x.dtype
    _, D = y.shape 

    noise_var = _resolve_noise_variance(model, device, dtype)
    const = D * (math.log(2 * math.pi) + torch.log(noise_var))

    x_mc = x.unsqueeze(0).expand(mc, *x.shape)
    y_pred = model(x_mc)
    if y_pred.dim() != 3:
        raise RuntimeError(
            f"Expected BNN output with MC dimension, got shape {tuple(y_pred.shape)}"
        )

    diff = y_pred - y.unsqueeze(0)                           # (mc, B, D)
    dist_norm = diff.pow(2).sum(dim=2) / noise_var           # (mc, B)
    logp = -0.5 * (dist_norm + const)                        # (mc, B)

    return logp



def mc_categorical_loglikelihood_samples(model, x, y, mc=250):
    r"""
        Monte Carlo samples of the log-likelihood.
        The model is expected to return logits; a log-softmax converts them
        into normalized log probabilities before selecting the target class.
    """
    if y.dim() != 1:
        y = y.view(-1)
    y = y.to(device=x.device, dtype=torch.long)

    x_mc = x.unsqueeze(0).expand(mc, *x.shape)
    logits = model(x_mc)
    if logits.dim() != 3:
        raise RuntimeError(
            f"Expected BNN output with MC dimension, got shape {tuple(logits.shape)}"
        )
    log_probs = torch.log_softmax(logits, dim=-1)
    target_idx = y.view(1, -1).expand(mc, -1)
    logp = log_probs.gather(dim=2, index=target_idx.unsqueeze(-1)).squeeze(-1)
    return logp

