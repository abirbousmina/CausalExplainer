"""
tgat.py  —  Version FINALE (basée sur v11 qui convergeait à AP=0.634)
=======================================================================

HISTORIQUE DES VERSIONS ET CE QUI A FONCTIONNÉ :

  v11 → Val AP=0.634, Test AP=0.568  ← SEULE VERSION QUI CONVERGEAIT
    node_dim=200, n_layers=4, dropout=0.1, alpha=0.9, state_dim=64
    neg=uniform, warmup=5ep@3e-3

  v12 → AP=0.50 (régressé)
    node_dim=128  ← TROP PETIT, cassé la capacité
    neg=temporal  ← FAUX NÉGATIFS MASSIFS, cassé la convergence

  v13/v14 → jamais testés correctement (fichiers pas copiés sur Drive)

CETTE VERSION : exactement v11 + degree-weighted neg sampling
  Seul changement vs v11 : le negative sampling est géré dans train_base_models
  Le modèle lui-même est identique à v11.

AJOUT V10.1 : fonction load_tgat_from_checkpoint_robust pour gérer
  les checkpoints sans 'node_embedding.weight' et avec msg_dim variable.
"""

import torch
import torch.nn as nn
import numpy as np

from models.tgn_model import LinkPredictor


# ─────────────────────────────────────────────────────────────────────────────
# Mémoire EMA légère
# ─────────────────────────────────────────────────────────────────────────────

class EMAMemory(nn.Module):
    """
    Mémoire Exponential Moving Average par nœud.
    state[v] ← α * state[v] + (1-α) * proj(msg)
    Brise la symétrie initiale → le gradient peut séparer les nœuds.
    """
    def __init__(self, num_nodes: int, state_dim: int,
                 msg_dim: int, alpha: float = 0.9):
        super().__init__()
        self.alpha = alpha
        self.register_buffer('state', torch.zeros(num_nodes, state_dim))
        self.msg_proj = nn.Sequential(
            nn.Linear(msg_dim, state_dim),
            nn.LayerNorm(state_dim),
            nn.Tanh(),
        )

    def get_state(self, node_ids: torch.Tensor) -> torch.Tensor:
        return self.state[node_ids].detach()

    @torch.no_grad()
    def update_state(self, node_ids: torch.Tensor, msg: torch.Tensor):
        new_s = self.msg_proj(msg.float())
        self.state[node_ids] = (self.alpha * self.state[node_ids]
                                + (1 - self.alpha) * new_s)

    @torch.no_grad()
    def reset_state(self):
        self.state.zero_()


# ─────────────────────────────────────────────────────────────────────────────
# Encodeur temporel
# ─────────────────────────────────────────────────────────────────────────────

class HarmonicTimeEncoder(nn.Module):
    def __init__(self, time_dim: int):
        super().__init__()
        freqs = torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim)).float()
        self.basis_freq = nn.Parameter(freqs)
        self.phase      = nn.Parameter(torch.zeros(time_dim))

    def forward(self, ts: torch.Tensor) -> torch.Tensor:
        return torch.cos(ts.float().view(-1, 1) * self.basis_freq.view(1, -1)
                         + self.phase.view(1, -1))


# ─────────────────────────────────────────────────────────────────────────────
# Bloc résiduel
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Encodeur de rôle directionnel
# ─────────────────────────────────────────────────────────────────────────────

class RoleEncoder(nn.Module):
    def __init__(self, node_dim: int, state_dim: int, time_dim: int,
                 msg_dim: int, n_blocks: int = 4, dropout: float = 0.1):
        super().__init__()
        in_dim = node_dim + state_dim + time_dim + msg_dim
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.GELU(),
        )
        self.blocks      = nn.ModuleList([ResidualBlock(node_dim, dropout)
                                          for _ in range(n_blocks)])
        self.output_norm = nn.LayerNorm(node_dim)

    def forward(self, node_emb, state, t_enc, msg):
        x = self.input_proj(torch.cat([node_emb, state, t_enc, msg], dim=-1))
        for block in self.blocks:
            x = block(x)
        return self.output_norm(x)


# ─────────────────────────────────────────────────────────────────────────────
# Predictor asymétrique
# ─────────────────────────────────────────────────────────────────────────────

class AsymmetricLinkPredictor(nn.Module):
    def __init__(self, in_channels: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.LayerNorm(in_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, in_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels // 2, 1),
        )

    def forward(self, z_src, z_dst):
        return self.net(torch.cat([z_src, z_dst], dim=-1)).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# RobustTGAT FINAL
# ─────────────────────────────────────────────────────────────────────────────

class RobustTGAT(nn.Module):
    """
    TGAT FINAL : TGN-TGAT Hybride avec mémoire EMA.

    Architecture identique à v11 (seule version qui convergeait sur UCI) :
      node_dim=200, state_dim=64, n_layers=4, dropout=0.1, alpha=0.9

    Deux encodeurs de rôle distincts (src/dst) pour graphe unipartite.
    Mémoire EMA qui donne un historique individuel à chaque nœud.
    Initialisation asymétrique pour casser la symétrie initiale.

    API uniforme avec RobustTGN :
      forward(src, dst, t, msg)         → (z_src, z_dst)
      update_memory(src, dst, t, msg)   → met à jour EMA
      get_embeddings(src, dst, t, msg)  → (z_src, z_dst)
      _encode(nodes, t, msg)            → alias négatifs (dst_role)
      reset_memory()                    → remet à zéro la mémoire
    """

    STATE_DIM = 64

    def __init__(self, node_features_dim: int, time_dim: int, num_nodes: int,
                 n_layers: int = 4, n_heads: int = 4,
                 dropout: float = 0.1, edge_dim: int = 0, **kwargs):
        super().__init__()

        self.node_dim  = node_features_dim
        self.time_dim  = time_dim
        self.msg_dim   = max(edge_dim, 1)
        self.state_dim = self.STATE_DIM

        self.time_encoder = HarmonicTimeEncoder(time_dim)

        self.src_memory = EMAMemory(num_nodes, self.state_dim, self.msg_dim, alpha=0.9)
        self.dst_memory = EMAMemory(num_nodes, self.state_dim, self.msg_dim, alpha=0.9)

        # Initialisation asymétrique — casse la symétrie initiale
        # src : grande norme → gradients forts au début
        # dst : petite norme → se déplace vers les vrais positifs
        self.src_embedding = nn.Embedding(num_nodes, node_features_dim)
        self.dst_embedding = nn.Embedding(num_nodes, node_features_dim)
        nn.init.normal_(self.src_embedding.weight, mean=0.0, std=0.5)
        nn.init.normal_(self.dst_embedding.weight, mean=0.0, std=0.1)

        self.src_encoder = RoleEncoder(node_features_dim, self.state_dim,
                                       time_dim, self.msg_dim, n_layers, dropout)
        self.dst_encoder = RoleEncoder(node_features_dim, self.state_dim,
                                       time_dim, self.msg_dim, n_layers, dropout)

        self.predictor = AsymmetricLinkPredictor(node_features_dim, dropout=dropout)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _get_msg(self, B, msg, device):
        if msg is not None and msg.size(-1) >= self.msg_dim:
            return msg.float()[:, :self.msg_dim]
        return torch.zeros(B, self.msg_dim, device=device)

    def _enc_src(self, nodes, t, msg):
        t_enc = self.time_encoder(t.float())
        return self.src_encoder(
            self.src_embedding(nodes),
            self.src_memory.get_state(nodes).to(nodes.device),
            t_enc, msg
        )

    def _enc_dst(self, nodes, t, msg):
        t_enc = self.time_encoder(t.float())
        return self.dst_encoder(
            self.dst_embedding(nodes),
            self.dst_memory.get_state(nodes).to(nodes.device),
            t_enc, msg
        )

    # ── API publique ──────────────────────────────────────────────────────────

    def forward(self, src, dst, t, msg=None):
        m = self._get_msg(src.size(0), msg, src.device)
        return self._enc_src(src, t, m), self._enc_dst(dst, t, m)

    def update_memory(self, src, dst, t, msg):
        m = self._get_msg(src.size(0), msg, src.device)
        with torch.no_grad():
            self.src_memory.update_state(src, m)
            self.dst_memory.update_state(dst, m)

    def reset_memory(self):
        self.src_memory.reset_state()
        self.dst_memory.reset_state()

    def get_embeddings(self, src, dst, t, msg=None):
        with torch.no_grad():
            m = self._get_msg(src.size(0), msg, src.device)
            z_src = torch.clamp(torch.nan_to_num(self._enc_src(src, t, m)), -10, 10)
            z_dst = torch.clamp(torch.nan_to_num(self._enc_dst(dst, t, m)), -10, 10)
        return z_src, z_dst

    def _encode(self, nodes, t, msg=None):
        """Alias pour les négatifs dans train_base_models (rôle dst)."""
        m = self._get_msg(nodes.size(0), msg, nodes.device)
        return self._enc_dst(nodes, t, m)


# ─────────────────────────────────────────────────────────────────────────────
# Chargeur robuste de TGAT
# ─────────────────────────────────────────────────────────────────────────────

def load_tgat_from_checkpoint_robust(path: str, num_nodes: int, device: torch.device) -> RobustTGAT:
    """
    Charge un checkpoint TGAT en détectant automatiquement les dimensions
    et le msg_dim. Gère les checkpoints sans 'node_embedding.weight' (ex: uci_msg)
    en utilisant 'src_embedding.weight' à la place.
    """
    sd = torch.load(path, map_location=device)

    # Détection de la clé d'embedding et de msg_dim
    if 'node_embedding.weight' in sd:
        node_dim = sd['node_embedding.weight'].shape[1]
        msg_dim = 1  # fallback, sera ajusté si les mémoires existent
        embed_key = 'node_embedding.weight'
    elif 'src_embedding.weight' in sd:
        node_dim = sd['src_embedding.weight'].shape[1]
        # Extraire msg_dim depuis src_memory.msg_proj.0.weight
        if 'src_memory.msg_proj.0.weight' in sd:
            msg_dim = sd['src_memory.msg_proj.0.weight'].shape[1]
        else:
            msg_dim = 1
        embed_key = 'src_embedding.weight'
    else:
        raise KeyError("Ni 'node_embedding.weight' ni 'src_embedding.weight' trouvés dans le checkpoint")

    time_dim = sd['time_encoder.basis_freq'].shape[0]
    n_layers = max(1, sum(1 for k in sd if k.startswith('attn_layers.') and k.endswith('.norm1.weight')))

    # Instanciation avec edge_dim = msg_dim
    model = RobustTGAT(node_features_dim=node_dim, time_dim=time_dim,
                       num_nodes=num_nodes, n_layers=n_layers,
                       edge_dim=msg_dim).to(device)

    # Chargement avec strict=False
    model.load_state_dict(sd, strict=False)

    # Extension du nombre de nœuds si nécessaire
    num_nodes_ckpt = sd[embed_key].shape[0]
    if num_nodes > num_nodes_ckpt:
        print(f"  → Extension embeddings TGAT : {num_nodes_ckpt} -> {num_nodes} nœuds")
        with torch.no_grad():
            new_embed = torch.zeros(num_nodes, node_dim, device=device)
            new_embed[:num_nodes_ckpt] = model.src_embedding.weight[:num_nodes_ckpt]
            if hasattr(model, 'node_embedding'):
                model.node_embedding.weight = nn.Parameter(new_embed)
            else:
                model.src_embedding.weight = nn.Parameter(new_embed)
                model.dst_embedding.weight = nn.Parameter(new_embed.clone())

    print(f"  ✓ TGAT chargé : node_dim={node_dim}, time_dim={time_dim}, n_layers={n_layers}, msg_dim={msg_dim}")
    return model