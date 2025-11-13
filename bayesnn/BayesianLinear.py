import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F

class BayesianLinear(Module):

    __constants__ = ['mu_prior', 'rho', 'bias', 'in_features', 'out_features']
    
    def __init__(self,in_features:int, out_features:int,
                mu_prior:float = 0.0, rho:float = 0.0,
                bias: bool = True, output_noise:bool = False,
                device = None, dtype = None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.output_noise = output_noise
        self.device = device 
        self.in_features = in_features
        self.out_features = out_features

        # Weight Parameters 
        self.mean_weights = Parameter(torch.empty(out_features, # mu
                                                  self.in_features,
                                                  **factory_kwargs)) 
        
        self.rho_weights = Parameter(torch.empty(out_features, # rho parameter for std to generate samples
                                                 self.in_features,
                                                 **factory_kwargs)) 
        
        self.register_buffer("mean_prior", torch.tensor(float(mu_prior)))
        self.register_buffer("rho_prior", torch.tensor(float(rho)))

        # Bias 
        self.bias = bias
        if self.bias:
            self.mean_bias = Parameter(torch.empty(self.out_features, **factory_kwargs))
            self.rho_bias = Parameter(torch.empty(self.out_features, **factory_kwargs))
        else:
            self.register_parameter("mean_bias", None)
            self.register_parameter("rho_bias", None)

        self.initialize_params()

    def initialize_params(self):
        with torch.no_grad():
            # Initialize the mean and rho of the weights 
            self.mean_weights.fill_(self.mean_prior)
            self.rho_weights.fill_(self.rho_prior)

            # Initialize the mean and rho of the bias terms
            if self.bias :
                self.mean_bias.fill_(self.mean_prior)
                self.rho_bias.fill_(self.rho_prior)
        

    def _sample_weight(self, mc_samples=None):
        sigma_w = F.softplus(self.rho_weights).clamp_min(1e-6)
        if mc_samples is None:
            epsilon_w = torch.randn_like(self.rho_weights)
            return sigma_w.mul(epsilon_w).add(self.mean_weights)

        shape = (mc_samples,) + sigma_w.shape
        epsilon_w = torch.randn(shape, device=sigma_w.device, dtype=sigma_w.dtype)
        return sigma_w.unsqueeze(0).mul(epsilon_w).add(self.mean_weights.unsqueeze(0))

    def _sample_bias(self, mc_samples=None):
        if not self.bias:
            return None
        sigma_b = F.softplus(self.rho_bias).clamp_min(1e-6)
        if mc_samples is None:
            epsilon_b = torch.randn_like(self.mean_bias)
            return sigma_b.mul(epsilon_b).add(self.mean_bias)

        shape = (mc_samples,) + sigma_b.shape
        epsilon_b = torch.randn(shape, device=sigma_b.device, dtype=sigma_b.dtype)
        return sigma_b.unsqueeze(0).mul(epsilon_b).add(self.mean_bias.unsqueeze(0))

    def forward(self, x):
        if x.dim() == 2: # Forward method will call this 
            w = self._sample_weight()
            b = self._sample_bias()
            return F.linear(x, w, b)

        if x.dim() >= 3: # Loss will use this, we vectorize the mc x batch x out_dim
            mc_samples = x.shape[0]
            if x.shape[-1] != self.in_features:
                raise ValueError(
                    f"Expected input dim {self.in_features}, got {x.shape[-1]}"
                )
            x_flat = x.reshape(mc_samples, -1, self.in_features)
            w = self._sample_weight(mc_samples)
            b = self._sample_bias(mc_samples)
            out = torch.einsum("sbi,soi->sbo", x_flat, w)
            if b is not None:
                out = out + b.unsqueeze(1)
            return out.reshape(*x.shape[:-1], self.out_features)

        raise ValueError("BayesianLinear expects rank >= 2 inputs.")
    
    def extra_repr(self)->str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias}"
