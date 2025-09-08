# Forked from https://github.com/hmohebbi/disentangling_representations, with some changes.

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


def unweighted_average(x, m):
    m = m.unsqueeze(-1).expand_as(x)
    effective_x = x * m
    sum_effective_x = torch.sum(effective_x, dim=1)
    pooled_x = sum_effective_x / torch.sum(m, dim=1)
    return pooled_x


class SelfAttention(torch.nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = torch.nn.Parameter(torch.ones(1, embed_dim), requires_grad=True)

    def forward(self, x, attention_mask=None, output_attentions=False):
        """Input shape: Batch x Time x Hidden Dim"""

       # (batch_size, seq_len)
        scores = torch.matmul(x, self.query.unsqueeze(0).transpose(-2, -1)).squeeze(-1)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == False, float('-inf'))

        attn_weights = torch.nn.functional.softmax(scores, dim=-1).unsqueeze(-1)

        # Weighted average
        pooled_x = torch.sum(attn_weights * x, dim=1)

        outputs = (pooled_x, attn_weights) if output_attentions else (pooled_x,)
        return outputs


@dataclass
class VIBConfig():
    input_dim: Optional[int] = None
    latent_dim: Optional[int] = None
    num_classes: Optional[int] = None
    stage: Optional[str] = None
    layer_weight_averaging: Optional[bool] = False
    num_layers: Optional[int] = None
    stage_1_tasks: Optional[list] = None


class VariationalEncoder(torch.nn.Module):
    def __init__(self, config):
        super(VariationalEncoder, self).__init__()
        self.enc1 = torch.nn.Linear(config.input_dim, config.input_dim)
        self.enc2 = torch.nn.Linear(config.input_dim, config.input_dim)
        self.mu = torch.nn.Linear(config.input_dim, config.latent_dim)
        self.var = torch.nn.Linear(config.input_dim, config.latent_dim)

    def forward(self, h):
        o = F.gelu(self.enc1(h))
        o = F.gelu(self.enc2(o))

        mu = self.mu(o)
        var = F.softplus(self.var(o))  # To generate positive values

        return mu, var


class UtteranceDecoder(torch.nn.Module):
    def __init__(self, config):
        super(UtteranceDecoder, self).__init__()
        if config.stage_1_tasks is not None:
            self.clf = torch.nn.Linear(
                config.latent_dim * (1 + len(config.stage_1_tasks)), config.num_classes)
            # latent_dim depends on how many stage 1 tasks are concatenated
        else:
            self.clf = torch.nn.Linear(config.latent_dim, config.num_classes)

    def forward(self, z, m, cond=None):
        if cond == None:  # Stage 1
            logits = self.clf(z)
        else:  # Stage 2
            cond = torch.concat(cond, dim=-1)  # cond is received as list of representations
            concatenated = torch.concat([cond, z], dim=-1)

            logits = self.clf(concatenated)

        return logits


class VIB(torch.nn.Module):
    def __init__(self, config):
        super(VIB, self).__init__()
        self.layer_weight_averaging = config.layer_weight_averaging
        if self.layer_weight_averaging:
            self.layer_weights = torch.nn.Parameter(torch.ones(
                config.num_layers)/config.num_layers, requires_grad=True)

        self.encoder = VariationalEncoder(config)

        self.decoder = UtteranceDecoder(config)

    def forward(self, h, m=None, cond=None, output_attentions=False, noise=True, encoder_only=False):
        if self.layer_weight_averaging:
            # Compute weighted sum over layers
            w = torch.nn.functional.softmax(self.layer_weights, dim=0)
            h = torch.sum(h * w.view(1, w.shape[0], 1, 1), dim=1)

        mu, var = self.encoder(h)
        std = var ** 0.5
        # Reparameterization trick: introducing epsilon only
        # during training, and use the z = mu during inference
        if self.training and noise:
            eps = torch.randn_like(std)  # sample from N(0, 1)
            z = mu + std * eps
        else:
            z = mu

        if encoder_only == False:
            # Decoding
            logits = self.decoder(z, m, cond)
        else:
            logits = None

        return logits, mu, var
