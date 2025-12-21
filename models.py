import torch
import torch.nn.functional as F
from torch.nn import Linear
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean, LinearMean, ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
import json
with open('parameters.json', 'r') as json_file:
    config = json.load(json_file)

# output_dim_model = config["output_dim"]
nr_of_mixtures = config["nr_of_mixtures"]

class SimpleTransformerModel(torch.nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, output_dim):
        super(SimpleTransformerModel, self).__init__()
        self.transformer_encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            activation='relu'
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            self.transformer_encoder_layer, num_layers=3)
        self.output_layer = Linear(input_dim, output_dim)

    def forward(self, src, src_key_padding_mask=None):
        output = self.transformer_encoder(
            src, src_key_padding_mask=src_key_padding_mask)
        output = self.output_layer(output)
        return output


class SimpleGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean_module = ConstantMean(input_size=7), output_dim_model = 7):
        super(SimpleGPModel, self).__init__(train_x, train_y, likelihood)
        # self.mean_module = gpytorch.means.LinearMean(input_size=5)
        # self.mean_module = LinearMean(input_size=7)
        # self.mean_module = LinearMean(input_size=7)
        self.mean_module = mean_module
        self.output_dim_model = output_dim_model
        print("output dim model:", self.output_dim_model, "within model")
        # self.mean_module = ZeroMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.SpectralMixtureKernel(
        #     num_mixtures=4, ard_num_dims=self.output_dim_model)) # + gpytorch.kernels.RBFKernel(ard_num_dims=7)
        self.covar_module = ScaleKernel(gpytorch.kernels.RBFKernel(
            ard_num_dims=self.output_dim_model))
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.SpectralMixtureKernel(
        #     num_mixtures=nr_of_mixtures, ard_num_dims=self.output_dim_model)) # + gpytorch.kernels.RBFKernel(ard_num_dims=7)
        # self.covar_module = ScaleKernel(gpytorch.kernels.MaternKernel(
        #     nu=2.5, ard_num_dims=7))
        # self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=7))
        # self.covar_module =gpytorch.kernels.SpectralMixtureKernel(
        #     num_mixtures=4, ard_num_dims=7)
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
