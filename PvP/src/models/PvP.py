import torch
import torch.nn as nn
from torch.nn import init
from torch_geometric.utils import to_dense_batch
from transformers import AutoModel

from mind.main.batch import MINDBatch, ContentsEncoded


import torch.nn.functional as F

def init_weights(m: nn.Module):
    if isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight.data)

    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.zeros_(m.bias)

    if isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()


def is_precomputed(x):
    return type(x) is torch.Tensor

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256, dropout: float = 0.1):
        super().__init__()

        hidden_dim = multiple_of * ((2 * hidden_dim // 3 + multiple_of - 1) // multiple_of)

        self.proj = nn.Linear(dim, 2 * hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_proj = self.proj(x)
        x1, x2 = x_proj.chunk(2, dim=-1)


        output = F.silu(x1) * x2


        if self.out_proj is not None:
            output = self.out_proj(self.dropout(output))
        return output

class MoEAdaptorLayer(nn.Module):

    def __init__(self, n_exps, layers, dropout=0.3, noise=True):
        super(MoEAdaptorLayer, self).__init__()

        self.n_exps = n_exps
        self.noisy_gating = noise

        self.experts = nn.ModuleList([SwiGLU(layers[0], layers[1], 256,dropout) for i in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2)


class AdditiveAttention(nn.Module):
    def __init__(self, dim=100, r=2.):
        super().__init__()
        intermediate = int(dim * r)
        self.attn = nn.Sequential(
            nn.Linear(dim, intermediate),
            nn.Dropout(0.01),
            nn.LayerNorm(intermediate),
            nn.SiLU(),
            nn.Linear(intermediate, 1),
            nn.Softmax(1),
        )
        self.attn.apply(init_weights)

    def forward(self, context):
        w = self.attn(context).squeeze(-1)

        return torch.bmm(w.unsqueeze(1), context).squeeze(1), w

class ContentsEncoder(nn.Module):
    def __init__(
            self,
            pretrained_model_name: str,
    ):
        super().__init__()
        bert = AutoModel.from_pretrained(pretrained_model_name)
        self.dim = bert.config.hidden_size

        self.bert = bert
        #我加的
        self.moe_adaptor = MoEAdaptorLayer(
            4,
            [self.dim,4*self.dim],
            0.2
        )

        self.pooler = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.Dropout(0.01),
            nn.LayerNorm(self.dim),
            nn.SiLU(),
        )

        self.pooler.apply(init_weights)

    def forward(self, inputs: ContentsEncoded):
        x_t = self.bert(**inputs['title'])[0]

        x_t = self.moe_adaptor(x_t)
        x_t = x_t[:, 0]

        x = self.pooler(x_t)

        return x


class PvP(nn.Module):
    def __init__(
            self,
            pretrained_model_name: str,
            sa_pretrained_model_name: str,
    ):
        super(PvP, self).__init__()
        self.encoder = ContentsEncoder(pretrained_model_name)

        dim = self.encoder.dim

        bert = AutoModel.from_pretrained(sa_pretrained_model_name)
        self.self_attn = bert.transformer.layer[-1]

        self.additive_attn = AdditiveAttention(dim, 0.5)
    def forward(self, inputs: MINDBatch):
        if is_precomputed(inputs['x_hist']):
            x_hist = inputs['x_hist']
        else:
            x_hist = self.encoder.forward(inputs['x_hist'])
        x_hist, mask_hist = to_dense_batch(x_hist, inputs['batch_hist'])
        x_hist = self.self_attn.forward(x_hist, attn_mask=mask_hist)[0]

        x_hist, _ = self.additive_attn(x_hist)

        if is_precomputed(inputs['x_cand']):
            x_cand = inputs['x_cand']
        else:
            x_cand = self.encoder.forward(inputs['x_cand'])
        x_cand, mask_cand = to_dense_batch(x_cand, inputs['batch_cand'])

        logits = torch.bmm(x_hist.unsqueeze(1), x_cand.permute(0, 2, 1)).squeeze(1)
        logits = logits[mask_cand]

        targets = inputs['targets']
        if targets is None:
            return logits

        if self.training:
            criterion = nn.CrossEntropyLoss()

            loss = criterion(logits.reshape(targets.size(0), -1), targets)
        else:

            with torch.no_grad():
                criterion = nn.BCEWithLogitsLoss()
                loss = criterion(logits, targets.float())
        return loss, logits

