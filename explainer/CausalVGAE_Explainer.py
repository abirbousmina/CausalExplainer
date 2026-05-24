"""
CausalVGAE_Explainer.py  
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalVGAE_Explainer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 64):
        super().__init__()
        self.input_dim  = input_dim
        self.latent_dim = latent_dim

        # ── Réseau de masque SOFT ──────────────────────────────────────────
        # Produit un masque continu ∈ (0,1)^input_dim
        self.mask_net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )
        # Initialisation : biais négatif pour que le masque commence "sparse"
        # et force le modèle à ne sélectionner que les traits les plus saillants.
        nn.init.constant_(self.mask_net[-1].bias, -1.0)

        # ── Encodeurs VAE PRISME LATENT ────────────────────────────────────
        # Utilisés uniquement pour calculer les métriques d'indépendance
        self.enc_causal = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.enc_spurious = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.fc_mu_c = nn.Linear(hidden_dim, latent_dim)
        self.fc_lv_c = nn.Linear(hidden_dim, latent_dim)
        self.fc_mu_p = nn.Linear(hidden_dim, latent_dim)
        self.fc_lv_p = nn.Linear(hidden_dim, latent_dim)

        # ── Baseline FIXE (mean global du training set) ────────────────────
        self.register_buffer('baseline', torch.zeros(1, input_dim))
        self.register_buffer('baseline_set', torch.tensor(False))

    @torch.no_grad()
    def set_baseline(self, all_z: torch.Tensor):
        """Pose la baseline = mean global du training set."""
        self.baseline.data = all_z.mean(dim=0, keepdim=True).detach()
        self.baseline_set.data = torch.tensor(True)

    def _reparameterize(self, mu, lv):
        std = (0.5 * lv.clamp(-5, 2)).exp()
        return mu + torch.randn_like(std) * std

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor,
                adjacency_exists: torch.Tensor = None, temperature: float = 1.0):
        B = z_src.size(0)

        # ── Masque soft ────────────────────────────────────────────────
        logits_mask = self.mask_net(torch.cat([z_src, z_dst], dim=-1))
        
        if self.training:
            # Gumbel-Softmax : température basse (0.1) pour un comportement binaire
            noise  = torch.zeros_like(logits_mask).uniform_(1e-6, 1 - 1e-6)
            gumbel = -torch.log(-torch.log(noise))
            mask = torch.sigmoid((logits_mask + gumbel) / max(temperature, 0.1))
        else:
            mask = torch.sigmoid(logits_mask)

        self._mask_dim = mask
        bsl = self.baseline.expand(B, -1) if bool(self.baseline_set) else torch.zeros_like(z_src)

        # ── z_causal et z_spurious (IN-DISTRIBUTION) ───────────────────
        self._z_c_s = z_src * mask + bsl * (1.0 - mask)
        self._z_c_d = z_dst * mask + bsl * (1.0 - mask)
        self._z_p_s = z_src * (1.0 - mask) + bsl * mask
        self._z_p_d = z_dst * (1.0 - mask) + bsl * mask

        # ── PRISME latent (pour métriques d'indépendance) ──────────────
        hc_s = self.enc_causal(self._z_c_s)
        hp_s = self.enc_spurious(self._z_p_s)
        
        self._mu_c, self._lv_c = self.fc_mu_c(hc_s), self.fc_lv_c(hc_s)
        self._mu_p, self._lv_p = self.fc_mu_p(hp_s), self.fc_lv_p(hp_s)
        
        zc = self._reparameterize(self._mu_c, self._lv_c)
        zp = self._reparameterize(self._mu_p, self._lv_p)

        # ── Sorties ────────────────────────────────────────────────────
        mask_scalar = mask.mean(dim=-1)
        if adjacency_exists is not None:
            mask_scalar = mask_scalar * adjacency_exists
        logits = torch.stack([1 - mask_scalar, mask_scalar], dim=-1)

        return mask_scalar, logits, (zc, zc), (zp, zp)

    def _full_loss(self, base_model, z_src, z_dst, y_orig, zc, zp,
                   epoch=1, target_spa=0.15, lambda_cfx=2.0):
        y_bin = (y_orig >= 0.5).float()

        # ── L_fid : Suffisance (z_causal reproduit y_orig) ──────────────
        y_causal = torch.sigmoid(base_model.predictor(self._z_c_s, self._z_c_d).squeeze(-1))
        L_fid = F.binary_cross_entropy(y_causal.clamp(1e-7, 1-1e-7), y_orig.detach().clamp(1e-7, 1-1e-7))

        # ── L_cfx : Nécessité (z_spurious inverse la prédiction) ────────
        y_spurious = torch.sigmoid(base_model.predictor(self._z_p_s, self._z_p_d).squeeze(-1))
        L_cfx = F.binary_cross_entropy(y_spurious.clamp(1e-7, 1-1e-7), (1.0 - y_bin).detach().clamp(1e-7, 1-1e-7))

        # ── L_spa : Sparsité (MSE douce) ─────────────────────────────────
        # La MSE permet au gradient de construire une hiérarchie de poids
        spa = self._mask_dim.mean()
        L_spa = F.mse_loss(spa, torch.tensor(target_spa, device=spa.device)) * 5.0

        # ── L_ind : Disentanglement (Zc et Zp orthogonaux) ───────────────
        L_ind = F.cosine_similarity(self._mu_c, self._mu_p, dim=-1).abs().mean()

        # ── L_kl : Régularisation VAE ────────────────────────────────────
        L_kl = -0.5 * torch.mean(1 + self._lv_c - self._mu_c.pow(2) - self._lv_c.exp())

        # ── Loss totale ──────────────────────────────────────────────────
        loss = L_fid + (lambda_cfx * L_cfx) + L_spa + (0.5 * L_ind) + (0.01 * L_kl)

        return loss, L_fid, L_cfx, L_spa, L_ind

    def loss_function(self, logits, base_model, z_src, z_dst, y_orig, zc_pair, zp_pair, prev_logits=None):
        total, L_fid, L_cfx, L_spa, L_ind = self._full_loss(base_model, z_src, z_dst, y_orig, zc_pair, zp_pair, epoch=1, target_spa=0.15, lambda_cfx=2.0)
        return total, L_fid, L_ind, L_spa