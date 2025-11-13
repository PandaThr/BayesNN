import math

from torch.nn import Module
from . import functionals as F
import torch.nn.functional as FT
import torch


class _Loss(Module):
    def __init__(self, model: Module, likelihood: str = "gaussian"):
        super().__init__()
        self.model = model
        self.likelihood = likelihood
        self._bayesian_modules = tuple(self._collect_bayesian_modules()) # Cache the layers
        self._loglikelihood_samples = self._select_loglikelihood_fn(likelihood)

    def _collect_bayesian_modules(self):
        modules = []
        for module in self.model.modules():
            if hasattr(module, "mean_weights") and hasattr(module, "rho_weights"):
                modules.append(module)
        return modules

    @staticmethod
    def _select_loglikelihood_fn(likelihood: str):
        if likelihood == "gaussian":
            return F.mc_loglikelihood_samples
        if likelihood == "categorical":
            if not hasattr(F, "mc_categorical_loglikelihood_samples"):
                raise AttributeError("Categorical likelihood requested but helper is missing.")
            return F.mc_categorical_loglikelihood_samples
        raise ValueError(f"Unsupported likelihood '{likelihood}'.")

    def kl_divergence(self):
        kl_terms = []
        for module in self._bayesian_modules:
            posterior_std = FT.softplus(module.rho_weights).clamp_min(1e-6)
            prior_std = FT.softplus(module.rho_prior).clamp_min(1e-6)
            kl_terms.append(
                F.bayesian_kl_div(module.mean_weights, posterior_std, module.mean_prior, prior_std)
            )

            if getattr(module, "mean_bias", None) is not None and getattr(module, "rho_bias", None) is not None:
                posterior_std_bias = FT.softplus(module.rho_bias).clamp_min(1e-6)
                prior_std_bias = FT.softplus(module.rho_prior).clamp_min(1e-6)
                kl_terms.append(
                    F.bayesian_kl_div(module.mean_bias, posterior_std_bias, module.mean_prior, prior_std_bias)
                )

        if not kl_terms:
            ref_param = next(self.model.parameters(), None)
            device = ref_param.device if ref_param is not None else torch.device("cpu")
            return torch.zeros((), device=device)
        return torch.stack(kl_terms).sum()


class BNN_VBLoss(_Loss):
    r"""
    Monte Carlo Variational Bayes loss.

    A Bayesian loss where the priors on each weight's \mu_{j,k} and \rho_{j,k} determine the update on the posterior,
    which is refined jointly by data and prior information. It computes a variational Bayes objective, using Monte Carlo
    sampling to approximate the likelihood. Loss includes KL divergence for two Gaussians.

    Parameters
    ----------
    model : torch.nn.Module
        Bayesian network whose posterior we regularize.
    N : int
        Dataset size (used to scale the log-likelihood term).
    mc : int, optional
        Number of Monte Carlo samples for the marginal log-likelihood estimate.
    beta : float, optional
        Weight on the KL divergence term.
    return_components : bool, optional
        When True, forward returns a tuple (total_loss, neg_loglike, kl) to allow logging.
    """
    
    def __init__(
        self,
        model: Module,
        N: int,
        mc: int = 50,
        beta: float = 1.0,
        return_components: bool = False,
        likelihood: str = "gaussian",
    ):
        super().__init__(model, likelihood=likelihood)
        self.mc = mc
        self.N = N
        self.beta = beta
        self.return_components = return_components

    def forward(self, x, y):
        logp_samples = self._loglikelihood_samples(self.model, x, y, mc=self.mc)
        neg_loglike_mc = -self.N * logp_samples.mean()
        kl = self.kl_divergence()
        total = neg_loglike_mc + self.beta * kl
        if self.return_components:
            return total, neg_loglike_mc.detach(), kl.detach()
        return total  


class BNN_AlphaDivergenceLoss(_Loss):
    r"""
    Alpha-divergence objective following Yingzhen Li and Yarin Gal. "Dropout inference in Bayesian neural networks with alpha-
    divergences", (7).

    The per-batch loss is:
        L_\alpha^{MC} = - (N / |B|) * (1/\alpha) \sum_{i \in B} \log \left( \frac{1}{mc} \sum_{k=1}^{mc} p(y_i | x_i, W_k, \Sigma)^\alpha \right)
        + \beta * KL(q(\mu,\rho) || p_0)
    where W_k come from the reparameterized BNN.

    return_components : bool, optional
        If True, forward returns (total_loss, neg_loglike_component, kl_component) for logging.
    """

    def __init__(
        self,
        model: Module,
        N: int,
        alpha: float = 0.5,
        mc: int = 50,
        beta: float = 1.0,
        alpha_eps: float = 1e-6,
        return_components: bool = False,
        likelihood: str = "gaussian",
    ):
        super().__init__(model, likelihood=likelihood)
        if alpha <= 0.0:
            raise ValueError("alpha must be positive for the alpha-divergence loss.")
        self.N = N
        self.alpha = alpha
        self.mc = mc
        self.beta = beta
        self.alpha_eps = alpha_eps # Numerical stability
        self._log_mc = math.log(self.mc)
        self.return_components = return_components

    def forward(self, x, y):
        if self.alpha < self.alpha_eps: # For numerical stability! alpha->0 \alpha_divergence->VB
            logp_samples = self._loglikelihood_samples(self.model, x, y, mc=self.mc)
            neg_loglike_mc = -self.N * logp_samples.mean()
            kl = self.kl_divergence()
            total = neg_loglike_mc + self.beta * kl
            if self.return_components:
                return total, neg_loglike_mc.detach(), kl.detach()
            return total

        logp_samples = self._loglikelihood_samples(self.model, x, y, mc=self.mc)  # (mc, B)
        log_mean_exp = torch.logsumexp(self.alpha * logp_samples, dim=0) - self._log_mc
        batch_objective = log_mean_exp.mean() / self.alpha
        neg_loglike_alpha = -self.N * batch_objective
        kl = self.kl_divergence()
        total = neg_loglike_alpha + self.beta * kl
        if self.return_components:
            return total, neg_loglike_alpha.detach(), kl.detach()
        return total
       
