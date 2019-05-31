# %%
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn import functional as F


class VariationalAutoEncoder(nn.Module):
    def __init__(
        self,
        data_dim,
        latent_dim=32,
        enc_dims=[128],
        dec_dims=[128],
        layer_type="fc",
        dec_activation=None,
    ):
        super(VariationalAutoEncoder, self).__init__()
        self.enc_layers, self.dec_layers = self._build_enc_dec(
            data_dim=data_dim,
            latent_dim=latent_dim,
            enc_dims=enc_dims,
            dec_dims=dec_dims,
            layer_type=layer_type,
            dec_activation=dec_activation,
        )

    def _build_enc_dec(
        self, data_dim, latent_dim, enc_dims, dec_dims, layer_type, dec_activation
    ):
        enc_layers = self._build_layers(
            dims=[data_dim] + enc_dims + [latent_dim],
            layer_type=layer_type,
            last_layer_activation=None,
            mu_var=True,
        )

        dec_layers = self._build_layers(
            dims=[latent_dim] + dec_dims + [data_dim],
            layer_type=layer_type,
            last_layer_activation=dec_activation,
            mu_var=False,
        )
        return enc_layers, dec_layers

    @staticmethod
    def _build_layers(dims, layer_type, last_layer_activation, mu_var):
        layers = []
        if layer_type == "fc":
            layer = nn.Linear

        for i, (in_dim, out_dim) in enumerate(zip(dims[:-2], dims[1:-1])):
            l = nn.Sequential(layer(in_dim, out_dim), nn.ReLU())
            layers.append(l)

        n_last_layer = 2 if mu_var else 1
        for _ in range(n_last_layer):
            if last_layer_activation:
                last_layer = nn.Sequential(
                    layer(dims[-2], dims[-1]), last_layer_activation
                )
            else:
                last_layer = layer(dims[-2], dims[-1])
            layers.append(last_layer)

        return nn.ModuleList(layers)

    def encode(self, x):
        for layer in self.enc_layers[:-2]:
            x = layer(x)
        mu = self.enc_layers[-2](x)
        log_var = self.enc_layers[-1](x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z

    def decode(self, z):
        for layer in self.dec_layers[:-1]:
            z = layer(z)
        dec_output = self.dec_layers[-1](z)
        return dec_output

    def forward(self, x):
        z_mu, z_log_var = self.encode(x)
        if self.training:
            z = self.reparameterize(z_mu, z_log_var)
        else:
            z = z_mu
        x_recon = self.decode(z)
        return z_mu, z_log_var, x_recon


class ConditionalVariationalAutoEncoder(VariationalAutoEncoder):
    def __init__(self, **kwargs):
        n_condition = kwargs.pop("n_condition")
        super(ConditionalVariationalAutoEncoder, self).__init__(**kwargs)
        
        kwargs["data_dim"] += n_condition
        self.enc_layers, _ = self._build_enc_dec(**kwargs)

        kwargs["latent_dim"] += n_condition
        kwargs["data_dim"] -= n_condition
        _, self.dec_layers = self._build_enc_dec(**kwargs)

    def forward(self, x):
        assert len(x) == 2
        x, y = x
        x = torch.cat([x, y], dim=-1)
        z_mu, z_log_var = self.encode(x)
        if self.training:
            z = self.reparameterize(z_mu, z_log_var)
        else:
            z = z_mu
        z = torch.cat([z, y], dim=-1)
        x_recon = self.decode(z)
        return z_mu, z_log_var, x_recon


#%%
