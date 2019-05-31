# %%
import torch
import torch.nn as nn


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
        )

    def _build_enc_dec(self, data_dim, latent_dim, enc_dims, dec_dims, layer_type):
        enc_dims.insert(0, data_dim)
        enc_dims.append(latent_dim)
        enc_layers = self._build_layers(
            dims=enc_dims,
            layer_type=layer_type,
            last_layer_activation=None,
            mu_var=True,
        )

        dec_dims.append(data_dim)
        dec_dims.insert(0, latent_dim)
        dec_layers = self._build_layers(
            dims=dec_dims,
            layer_type=layer_type,
            last_layer_activation=nn.Sigmoid(),
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

    def sample(self, mu, log_var):
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
        if self.model.training:
            z = self.sample(z_mu, z_log_var)
        else:
            z = z_mu
        x_recon = self.decode(z)
        return z_mu, z_log_var, x_recon


# #%%
# params = {
#     "data_dim": 1011,
#     "latent_dim": 32,
#     "enc_dims": [128],
#     "dec_dims": [128],
#     "layer_type": "fc",
# }
# vae = VariationalAutoEncoder(**params)


#%%
