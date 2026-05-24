"""
tgn_model.py  —  Version 10.0
================================
AMÉLIORATION v10 pour UCI :

  Avec msg_dim=8 (features structurelles du data_loader v8),
  TGNMemory peut maintenant apprendre à distinguer les interactions.

  Le GRU reçoit : [msg(8) | memory_src(200) | time_enc(100)]
  Au lieu de    : [zeros(1) | memory_src(200) | time_enc(100)]

  Différence clé : msg=zeros → gradient nul sur la mémoire.
                   msg=8-dim structural → gradient riche → convergence.

  Pas d'autres changements architecturaux — TGNMemory + proj + predictor.
  La BatchSelfAttention reste sans leakage (src seul, dst seul).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator


# ─────────────────────────────────────────────────────────────────────────────
# LinkPredictor — logits bruts
# ─────────────────────────────────────────────────────────────────────────────

class LinkPredictor(nn.Module):
    def __init__(self, in_channels: int, dropout: float = 0.1):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.drop    = nn.Dropout(dropout)
        self.lin_out = Linear(in_channels, 1)

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        return self.lin_out(self.drop(h.relu())).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Auto-attention sur le batch — sans leakage src↔dst
# ─────────────────────────────────────────────────────────────────────────────

class BatchSelfAttention(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.q    = Linear(dim, dim, bias=False)
        self.k    = Linear(dim, dim, bias=False)
        self.v    = Linear(dim, dim, bias=False)
        self.o    = Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.scale = dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.q(x); K = self.k(x); V = self.v(x)
        attn = F.softmax(Q @ K.T * self.scale, dim=-1)
        out  = self.o(self.drop(attn) @ V)
        return self.norm(x + self.drop(out))


# ─────────────────────────────────────────────────────────────────────────────
# RobustTGN v10
# ─────────────────────────────────────────────────────────────────────────────

class RobustTGN(nn.Module):
    """
    TGN v10 : TGNMemory accepte edge_features_dim variable.

    Avec data_loader v8 :
      UCI/CollegeMsg → edge_features_dim=8 (features structurelles causales)
      Wikipedia/Reddit → edge_features_dim=172 (features JODIE originales)

    Architecture : TGNMemory → proj(2 couches) → BatchSelfAttention → LinkPredictor
    """

    def __init__(self, num_nodes: int, node_features_dim: int,
                 edge_features_dim: int, memory_dim: int, time_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.memory_dim      = memory_dim
        self.edge_features_dim = edge_features_dim

        self.memory = TGNMemory(
            num_nodes         = num_nodes,
            raw_msg_dim       = edge_features_dim,
            memory_dim        = memory_dim,
            time_dim          = time_dim,
            message_module    = IdentityMessage(edge_features_dim, memory_dim, time_dim),
            aggregator_module = LastAggregator(),
        )

        self.proj = nn.Sequential(
            Linear(memory_dim, memory_dim),
            nn.LayerNorm(memory_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            Linear(memory_dim, memory_dim),
            nn.LayerNorm(memory_dim),
        )

        self.batch_attn = BatchSelfAttention(memory_dim, dropout=dropout)
        self.predictor  = LinkPredictor(memory_dim, dropout=dropout)

    def _embed(self, nodes: torch.Tensor) -> torch.Tensor:
        z, _ = self.memory(nodes)
        z = self.proj(z)
        z = self.batch_attn(z)
        return z

    def forward(self, src: torch.Tensor, dst: torch.Tensor):
        """src et dst encodés SÉPARÉMENT — pas de leakage."""
        return self._embed(src), self._embed(dst)

    def update_memory(self, src, dst, t, msg):
        """
        Met à jour la mémoire avec msg.
        Avec msg_dim=8 : le GRU reçoit un signal riche → converge.
        Avec msg_dim=1 (zeros) : le GRU ne peut rien apprendre.
        """
        self.memory.update_state(src, dst, t, msg)

    def get_embeddings(self, src: torch.Tensor, dst: torch.Tensor,
                       t: torch.Tensor):
        with torch.no_grad():
            z_src = torch.clamp(torch.nan_to_num(self._embed(src)), -10, 10)
            z_dst = torch.clamp(torch.nan_to_num(self._embed(dst)), -10, 10)
        return z_src, z_dst


# ─────────────────────────────────────────────────────────────────────────────
# Auto-loaders depuis checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_tgn_from_checkpoint(path: str, num_nodes: int,
                              edge_dim: int, device: torch.device) -> RobustTGN:
    sd         = torch.load(path, map_location=device)
    memory_dim = sd['memory.memory'].shape[1]

    if 'memory.time_enc.lin.weight' in sd:
        time_dim = sd['memory.time_enc.lin.weight'].shape[0]
    else:
        gru_in   = sd['memory.gru.weight_ih'].shape[1]
        time_dim = gru_in - memory_dim - edge_dim

    model = RobustTGN(
        num_nodes=num_nodes, node_features_dim=memory_dim,
        edge_features_dim=edge_dim, memory_dim=memory_dim, time_dim=time_dim,
    ).to(device)
    model.load_state_dict(sd, strict=False)
    print(f"  TGN chargé : memory_dim={memory_dim}, time_dim={time_dim}, edge_dim={edge_dim}")
    return model


def load_tgat_from_checkpoint(path: str, num_nodes: int, device: torch.device):
    from models.tgat import RobustTGAT
    sd       = torch.load(path, map_location=device)
    node_dim = sd['node_embedding.weight'].shape[1]
    time_dim = sd['time_encoder.basis_freq'].shape[0]
    n_layers = max(
        sum(1 for k in sd if k.startswith('attn_layers.') and k.endswith('.norm1.weight')), 1
    )
    model = RobustTGAT(
        node_features_dim=node_dim, time_dim=time_dim,
        num_nodes=num_nodes, n_layers=n_layers,
    ).to(device)
    model.load_state_dict(sd, strict=False)
    print(f"  TGAT chargé : node_dim={node_dim}, n_layers={n_layers}")
    return model