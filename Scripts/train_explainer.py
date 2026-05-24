#!/usr/bin/env python3
"""
train_explainer.py 
============================================================================
"""

import os, sys, argparse, time, math, inspect, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = "/content/drive/MyDrive/causalvgae/CausalVGAE_Project"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.data_loader import get_dataset, temporal_train_val_test_split
from explainer.CausalVGAE_Explainer import CausalVGAE_Explainer


# ════════════════════════════════════════════════════════════════════════
# SEED
# ════════════════════════════════════════════════════════════════════════
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ════════════════════════════════════════════════════════════════════════
# Modèles de base
# ════════════════════════════════════════════════════════════════════════
class _LP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a = nn.Linear(d, d)
        self.b = nn.Linear(d, d)
        self.c = nn.Linear(d, 1)
    def forward(self, s, d):
        return self.c((self.a(s) + self.b(d)).relu()).squeeze(-1)

def _proj(sd, md):
    idl = {}
    mi = -1
    for k in sd:
        if k.startswith('proj.'):
            p = k.split('.')
            if len(p) >= 3 and p[1].isdigit() and p[2] == 'weight':
                i = int(p[1])
                w = sd[k]
                mi = max(mi, i)
                idl[i] = nn.Linear(w.shape[1], w.shape[0]) if w.dim() == 2 else nn.LayerNorm(w.shape[0])
    if not idl:
        return nn.Sequential(nn.Linear(md, md), nn.LayerNorm(md), nn.GELU())
    ls = []
    for i in range(mi + 2):
        ls.append(idl[i] if i in idl else nn.GELU())
    if isinstance(ls[-1], nn.Linear):
        ls.append(nn.GELU())
    return nn.Sequential(*ls)

class _TGN(nn.Module):
    def __init__(self, sd, n, ed, md, td):
        super().__init__()
        self.ed = ed
        self.memory = TGNMemory(n, ed, md, td, IdentityMessage(ed, md, td), LastAggregator())
        self.proj = _proj(sd, md)
        self.predictor = _LP(md)
    def _e(self, n):
        z, _ = self.memory(n)
        return self.proj(z)
    def forward(self, s, d):
        return self._e(s), self._e(d)
    def update_memory(self, s, d, t, m):
        ed = self.ed
        if m is None:
            m = torch.zeros(s.size(0), ed, device=s.device)
        elif m.size(-1) > ed:
            m = m[:, :ed].float()
        elif m.size(-1) < ed:
            m = torch.cat([m.float(), torch.zeros(m.size(0), ed - m.size(-1), device=m.device)], -1)
        else:
            m = m.float()
        self.memory.update_state(s, d, t, m)
    def reset_memory(self):
        self.memory.reset_state()
    def get_embeddings(self, s, d, t=None, m=None):
        with torch.no_grad():
            return (torch.clamp(torch.nan_to_num(self._e(s)), -10, 10),
                    torch.clamp(torch.nan_to_num(self._e(d)), -10, 10))

class _HTE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f = nn.Parameter(torch.from_numpy(1 / 10**np.linspace(0, 9, d)).float())
        self.p = nn.Parameter(torch.zeros(d))
    def forward(self, t):
        return torch.cos(t.float().view(-1, 1) * self.f.view(1, -1) + self.p.view(1, -1))

class _RB(nn.Module):
    def __init__(self, d, dr=0.1):
        super().__init__()
        self.n = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d*2), nn.GELU(), nn.Dropout(dr),
                               nn.Linear(d*2, d), nn.Dropout(dr))
    def forward(self, x):
        return x + self.n(x)

class _AP(nn.Module):
    def __init__(self, d, dr=0.1):
        super().__init__()
        self.n = nn.Sequential(nn.Linear(d*2, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dr),
                               nn.Linear(d, d//2), nn.GELU(), nn.Dropout(dr), nn.Linear(d//2, 1))
    def forward(self, s, d):
        return self.n(torch.cat([s, d], -1)).squeeze(-1)

class _TGAT(nn.Module):
    def __init__(self, sd, n):
        super().__init__()
        self._a = ('ema' if 'src_embedding.weight' in sd else
                   'att' if any(k.startswith('attn_layers.') for k in sd) else 'mlp')
        self._build(sd, n)
        self._build_pred(sd)
    def _build(self, sd, n):
        if self._a == 'ema':
            self._nd = sd['src_embedding.weight'].shape[1]
            self._td = sd['time_encoder.basis_freq'].shape[0]
            self._sd = sd['src_memory.state'].shape[1]
            self._md = sd['src_memory.msg_proj.0.weight'].shape[1]
            self.te = _HTE(self._td)
            self.se = nn.Embedding(n, self._nd)
            self.de = nn.Embedding(n, self._nd)
            self.register_buffer('ss', torch.zeros(n, self._sd))
            self.register_buffer('ds', torch.zeros(n, self._sd))
            nb = max(sum(1 for k in sd if k.startswith('src_encoder.blocks.') and k.endswith('.net.1.weight')), 1)
            id_ = self._nd + self._sd + self._td + self._md
            self.si = nn.Linear(id_, self._nd)
            self.di = nn.Linear(id_, self._nd)
            self.sb = nn.ModuleList([_RB(self._nd) for _ in range(nb)])
            self.db = nn.ModuleList([_RB(self._nd) for _ in range(nb)])
            self.sn = nn.LayerNorm(self._nd)
            self.dn = nn.LayerNorm(self._nd)
        else:
            self._nd = sd['node_embedding.weight'].shape[1]
            self._td = sd['time_encoder.basis_freq'].shape[0]
            self._md = 0
            self.te = _HTE(self._td)
            self.ne = nn.Embedding(n, self._nd)
            iw = sd.get('input_proj.0.weight')
            self._id = iw.shape[1] if iw is not None else self._nd + self._td
            self.ip = nn.Linear(self._id, self._nd)
            self.on = nn.LayerNorm(self._nd)
            if self._a == 'mlp':
                nb = max(sum(1 for k in sd if k.startswith('blocks.') and k.endswith('.net.1.weight')), 1)
                self.bk = nn.ModuleList([_RB(self._nd) for _ in range(nb)])
    def _build_pred(self, sd):
        for k in sd:
            if 'predictor' in k and 'lin_src' in k and 'weight' in k:
                self.predictor = _LP(sd[k].shape[0])
                return
            if 'predictor' in k and 'net.0.weight' in k:
                self.predictor = _AP(sd[k].shape[1]//2)
                return
        self.predictor = _LP(self._nd)
    def _ee(self, n, t, m, e, ip, bk, no, st):
        te = self.te(t.float())
        ne = e(n)
        s = st[n].detach().to(n.device)
        md = self._md
        mg = (m.float()[:, :md] if m is not None and m.size(-1) >= md else torch.zeros(n.size(0), md, device=n.device))
        x = torch.relu(ip(torch.cat([ne, s, te, mg], -1)))
        for b in bk:
            x = b(x)
        return no(x)
    def _es(self, n, t, m):
        te = self.te(t.float())
        ne = self.ne(n)
        c = torch.cat([ne, te], -1)
        if c.size(-1) < self._id:
            c = torch.cat([c, torch.zeros(c.size(0), self._id - c.size(-1), device=c.device)], -1)
        elif c.size(-1) > self._id:
            c = c[:, :self._id]
        x = torch.relu(self.ip(c))
        if self._a == 'mlp':
            for b in self.bk:
                x = b(x)
        return self.on(x)
    def forward(self, s, d, t, m=None):
        if self._a == 'ema':
            return (self._ee(s, t, m, self.se, self.si, self.sb, self.sn, self.ss),
                    self._ee(d, t, m, self.de, self.di, self.db, self.dn, self.ds))
        return self._es(s, t, m), self._es(d, t, m)
    def update_memory(self, *a): pass
    def reset_memory(self): pass
    def get_embeddings(self, s, d, t=None, m=None):
        t = t if t is not None else torch.zeros(s.size(0), device=s.device)
        with torch.no_grad():
            zs, zd = self.forward(s, d, t, m)
            return (torch.clamp(torch.nan_to_num(zs), -10, 10),
                    torch.clamp(torch.nan_to_num(zd), -10, 10))

def load_base_model(model_type, dataset_name, device):
    ckpt = f'{PROJECT_ROOT}/checkpoints/{model_type}_{dataset_name}_best.pth'
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Manquant : {ckpt}")
    sd = torch.load(ckpt, map_location=device)
    if model_type == 'tgn':
        md = sd['memory.memory'].shape[1]
        n = sd['memory.memory'].shape[0]
        td = sd['memory.time_enc.lin.weight'].shape[0] if 'memory.time_enc.lin.weight' in sd else 100
        gru = sd['memory.gru.weight_ih'].shape[1]
        ed = max(gru - 2*md - td, 1)
        model = _TGN(sd, n, ed, md, td)
        model.load_state_dict(sd, strict=False)
    else:
        n = sd['src_embedding.weight'].shape[0] if 'src_embedding.weight' in sd else sd['node_embedding.weight'].shape[0]
        model = _TGAT(sd, n)
        model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

def _get_embeddings(model, src, dst, t, msg=None):
    if hasattr(model, 'memory'):
        mx = model.memory.memory.size(0) - 1
        src = src.clamp(0, mx)
        dst = dst.clamp(0, mx)
    elif hasattr(model, 'ne'):
        mx = model.ne.weight.size(0) - 1
        src = src.clamp(0, mx)
        dst = dst.clamp(0, mx)
    elif hasattr(model, 'se'):
        mx = model.se.weight.size(0) - 1
        src = src.clamp(0, mx)
        dst = dst.clamp(0, mx)
    n_params = len(inspect.signature(model.get_embeddings).parameters)
    if n_params >= 5: return model.get_embeddings(src, dst, t, msg)
    elif n_params >= 4: return model.get_embeddings(src, dst, t)
    return model.get_embeddings(src, dst)

@torch.no_grad()
def _fill_memory(model, model_type, loader, device, reset_first=True):
    if model_type != 'tgn': return
    mx = model.memory.memory.size(0) - 1
    model.eval()
    if reset_first: model.memory.reset_state()
    for batch in loader:
        batch = batch.to(device)
        src, dst = batch.src.clamp(0, mx), batch.dst.clamp(0, mx)
        model.memory.update_state(src, dst, batch.t, batch.msg)
        model.memory.detach()

@torch.no_grad()
def compute_global_baseline(base_model, model_type, train_loader, device):
    _fill_memory(base_model, model_type, train_loader, device, reset_first=True)
    all_z = []
    for batch in train_loader:
        batch = batch.to(device)
        msg = batch.msg if batch.msg.size(1) > 1 else None
        z_src, z_dst = _get_embeddings(base_model, batch.src, batch.dst, batch.t, msg)
        all_z.append(z_src.cpu())
        all_z.append(z_dst.cpu())
        if model_type == 'tgn':
            base_model.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
            base_model.memory.detach()
    return torch.cat(all_z, dim=0).to(device)


# ════════════════════════════════════════════════════════════════════════
# Métriques (Avec Top-K Percentile pour courbes progressives)
# ════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def compute_metrics_with_curves(explainer, base_model, model_type, eval_loader, device,
                                update_memory=True, spa_thresholds=None, ablation='full'):
    if spa_thresholds is None:
        spa_thresholds = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    base_model.eval()
    explainer.eval()
    fp_by_spa = {s: [] for s in spa_thresholds}
    fm_by_spa = {s: [] for s in spa_thresholds}
    spa_actual_list, runtime_list, rf_list = [], [], []

    for batch in eval_loader:
        batch = batch.to(device)
        msg = batch.msg if batch.msg.size(1) > 1 else None
        start_time = time.time()

        z_src, z_dst = _get_embeddings(base_model, batch.src, batch.dst, batch.t, msg)
        y_orig = torch.sigmoid(base_model.predictor(z_src, z_dst).squeeze(-1))
        y_label = (y_orig >= 0.5).float()

        explainer(z_src, z_dst)
        mask = explainer._mask_dim
        
        # -------------------------------------------------------------
        # ABLATIONS D'ÉVALUATION (Masque Inversé ou Aléatoire)
        # -------------------------------------------------------------
        if ablation == 'spurious_only':
            mask = 1.0 - mask
        elif ablation == 'random_mask':
            mask = torch.rand_like(mask) # Génère du bruit uniforme [0, 1]
        # -------------------------------------------------------------

        baseline = explainer.baseline.expand(z_src.size(0), -1)
        
        spa_actual_list.extend(mask.mean(dim=-1).cpu().tolist())

        # Évaluation hiérarchique Top-K
        for thresh in spa_thresholds:
            if thresh == 0.0:
                mask_bin = torch.zeros_like(mask)
            else:
                k = max(1, int(thresh * mask.size(1)))
                val, _ = torch.topk(mask, k, dim=1)
                min_val = val[:, -1].unsqueeze(1)
                mask_bin = (mask >= min_val).float()

            z_causal = z_src * mask_bin + baseline * (1 - mask_bin)
            z_spur   = z_src * (1 - mask_bin) + baseline * mask_bin
            
            y_keep = torch.sigmoid(base_model.predictor(z_causal, z_causal).squeeze(-1))
            y_drop = torch.sigmoid(base_model.predictor(z_spur, z_spur).squeeze(-1))
            
            fm_by_spa[thresh].extend(((y_keep >= 0.5).float() == y_label).float().cpu().tolist())
            fp_by_spa[thresh].extend(((y_drop >= 0.5).float() != y_label).float().cpu().tolist())

        # RF à 0.20
        k20 = max(1, int(0.20 * mask.size(1)))
        val20, _ = torch.topk(mask, k20, dim=1)
        mb20 = (mask >= val20[:, -1].unsqueeze(1)).float()
        z_c20 = z_src * mb20 + baseline * (1 - mb20)
        
        rf_sum = 0.0
        for _ in range(5):
            noise = torch.randn_like(z_c20) * 0.05
            y_rf = torch.sigmoid(base_model.predictor(z_c20 + noise, z_c20 + noise).squeeze(-1))
            rf_sum += ((y_rf >= 0.5).float() == y_label).float().mean().item()
        rf_list.append(rf_sum / 5)
        
        runtime_list.append(time.time() - start_time)

        if model_type == 'tgn' and update_memory:
            mx = base_model.memory.memory.size(0) - 1
            base_model.memory.update_state(batch.src.clamp(0, mx), batch.dst.clamp(0, mx),
                                           batch.t, batch.msg)
            base_model.memory.detach()

    spa_vals = sorted(spa_thresholds)
    fp_means = [float(np.mean(fp_by_spa[s])) if fp_by_spa[s] else 0.0 for s in spa_vals]
    fm_means = [float(np.mean(fm_by_spa[s])) if fm_by_spa[s] else 0.0 for s in spa_vals]
    
    aufsc_plus = float(np.trapezoid(fp_means, spa_vals))
    aufsc_minus = float(np.trapezoid(fm_means, spa_vals))
    
    fp_20 = fp_means[spa_vals.index(0.20)]
    fm_20 = fm_means[spa_vals.index(0.20)]
    char = 2 * fp_20 * fm_20 / (fp_20 + fm_20 + 1e-9)

    return {
        'Fid+': fp_20, 'Fid-': fm_20, 'CHAR': char,
        'AUFSC+': aufsc_plus, 'AUFSC-': aufsc_minus,
        'RF': float(np.mean(rf_list)) if rf_list else 0.0,
        'Runtime': float(np.mean(runtime_list)) if runtime_list else 0.0,
        'Sparsity': float(np.mean(spa_actual_list)),
        'fp_curve': fp_means, 'fm_curve': fm_means,
    }


# ════════════════════════════════════════════════════════════════════════
# Entraînement
# ════════════════════════════════════════════════════════════════════════
DATASET_CONFIG = {
    'wikipedia':  {'target_spa': 0.20, 'lambda_cfx': 3.0, 'lr': 5e-4, 'patience': 12},
    'reddit':     {'target_spa': 0.20, 'lambda_cfx': 3.5, 'lr': 5e-4, 'patience': 12},
    'uci_msg':    {'target_spa': 0.20, 'lambda_cfx': 2.5, 'lr': 3e-4, 'patience': 10},
}
def get_cfg(ds): return DATASET_CONFIG.get(ds, {'target_spa': 0.20, 'lambda_cfx': 2.5, 'lr': 3e-4, 'patience': 5})

def run_training(model_type, dataset_name, ablation='full', seed=42, epochs=25,
                 target_spa_override=None, lambda_cfx_override=None,
                 lambda_ind_override=None, lambda_fid_override=None, suffix=''):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = get_cfg(dataset_name)
    target_spa = target_spa_override if target_spa_override is not None else cfg['target_spa']
    lambda_cfx = lambda_cfx_override if lambda_cfx_override is not None else cfg['lambda_cfx']

    lambda_ind = 0.0 if ablation == 'no_ind' else (lambda_ind_override if lambda_ind_override is not None else 0.5)
    default_lambda_ind = 0.5
    lambda_fid = lambda_fid_override if lambda_fid_override is not None else 1.0

    print(f"\n{'='*60}\n  EXPLAINER : {model_type.upper()} / {dataset_name.upper()} (Ablation: {ablation})\n{'='*60}")

    result = get_dataset(root_dir=f'{PROJECT_ROOT}/data', name=dataset_name)
    train_data, val_data, test_data = temporal_train_val_test_split(result.data)
    train_loader = TemporalDataLoader(train_data, batch_size=128, shuffle=False)
    val_loader   = TemporalDataLoader(val_data,   batch_size=128, shuffle=False)
    test_loader  = TemporalDataLoader(test_data,  batch_size=128, shuffle=False)

    base_model = load_base_model(model_type, dataset_name, device)

    with torch.no_grad():
        sb = next(iter(train_loader)).to(device)
        msg_s = sb.msg if sb.msg.size(1) > 1 else None
        zs, _ = _get_embeddings(base_model, sb.src, sb.dst, sb.t, msg_s)
        input_dim = zs.size(-1)

    explainer = CausalVGAE_Explainer(input_dim=input_dim, hidden_dim=128, latent_dim=64).to(device)
    
    all_z_train = compute_global_baseline(base_model, model_type, train_loader, device)
    explainer.set_baseline(all_z_train)

    optimizer = torch.optim.Adam(explainer.parameters(), lr=cfg['lr'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-5)

    # -------------------------------------------------------------
    # GESTION DES SAUVEGARDES (FIX ÉTUDE D'ABLATION)
    # Les ablations d'évaluation utilisent le fichier du modèle FULL.
    # -------------------------------------------------------------
    if ablation in ['spurious_only', 'random_mask']:
        max_ep = 0 # Pas d'entraînement !
        save_path = f'{PROJECT_ROOT}/checkpoints/explainer_{model_type}_{dataset_name}.pth'
    else:
        max_ep = epochs
        save_path = f'{PROJECT_ROOT}/checkpoints/explainer_{model_type}_{dataset_name}{suffix}.pth'
        
    best_char, pat = 0.0, 0

    for epoch in range(1, max_ep + 1):
        tau = max(0.1, 1.0 - (epoch - 1) / (max_ep - 1))
        explainer.train()
        if model_type == 'tgn': base_model.memory.reset_state()

        total_loss, nb = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Ep{epoch:02d}", leave=False)
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                msg = batch.msg if batch.msg.size(1) > 1 else None
                z_src, z_dst = _get_embeddings(base_model, batch.src, batch.dst, batch.t, msg)
                y_orig = torch.sigmoid(base_model.predictor(z_src, z_dst).squeeze(-1))
                if model_type == 'tgn':
                    base_model.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
                    base_model.memory.detach()

            _, _, zc, zp = explainer(z_src, z_dst, adjacency_exists=torch.ones(z_src.size(0), device=device), temperature=tau)

            loss, L_fid, L_cfx, L_spa, L_ind = explainer._full_loss(
                base_model, z_src, z_dst, y_orig, zc, zp, epoch=epoch, target_spa=target_spa, lambda_cfx=lambda_cfx
            )
            
            if lambda_ind != default_lambda_ind:
                loss = loss + (lambda_ind - default_lambda_ind) * L_ind
            if lambda_fid != 1.0:
                loss = loss - L_fid + (lambda_fid * L_fid)

            if torch.isnan(loss) or torch.isinf(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(explainer.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            nb += 1

        if nb == 0: continue

        if model_type == 'tgn': _fill_memory(base_model, model_type, train_loader, device, reset_first=True)
        val_metrics = compute_metrics_with_curves(explainer, base_model, model_type, val_loader, device, update_memory=True, ablation=ablation)
        scheduler.step(val_metrics['CHAR'])

        print(f"  Ep {epoch:02d} | Loss={total_loss/nb:.3f} | Val CHAR={val_metrics['CHAR']:.4f} (Fid+={val_metrics['Fid+']:.3f})")

        if val_metrics['CHAR'] > best_char:
            best_char = val_metrics['CHAR']
            pat = 0
            torch.save(explainer.state_dict(), save_path)
        else:
            pat += 1
            if pat >= cfg['patience']:
                break

    # Chargement du modèle
    try:
        explainer.load_state_dict(torch.load(save_path, map_location=device))
    except FileNotFoundError:
        print(f"⚠️ Erreur : {save_path} introuvable. Le modèle FULL doit être entraîné en premier !")

    if model_type == 'tgn':
        _fill_memory(base_model, model_type, train_loader, device, reset_first=True)
        _fill_memory(base_model, model_type, val_loader, device, reset_first=False)
        
    test_metrics = compute_metrics_with_curves(explainer, base_model, model_type, test_loader, device, update_memory=True, ablation=ablation)
    val_metrics = compute_metrics_with_curves(explainer, base_model, model_type, val_loader, device, update_memory=True, ablation=ablation)

    print(f"  TEST : CHAR={test_metrics['CHAR']:.4f} | Fid+={test_metrics['Fid+']:.4f} | Fid-={test_metrics['Fid-']:.4f}")

    base = {
        'Model': model_type.upper(), 'Dataset': dataset_name, 'Ablation': ablation, 'Seed': seed
    }
    
    test_row = {**base, 'Split': 'Test', **{k: test_metrics[k] for k in ['Fid+','Fid-','CHAR','AUFSC+','AUFSC-','RF','Runtime','Sparsity']}}
    val_row  = {**base, 'Split': 'Val',  **{k: val_metrics[k] for k in ['Fid+','Fid-','CHAR','AUFSC+','AUFSC-','RF','Runtime','Sparsity']}}

    spa_vals = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    for i, sv in enumerate(spa_vals):
        test_row[f'Fid+_{int(sv*100)}'] = test_metrics['fp_curve'][i]
        test_row[f'Fid-_{int(sv*100)}'] = test_metrics['fm_curve'][i]

    return test_row, val_row