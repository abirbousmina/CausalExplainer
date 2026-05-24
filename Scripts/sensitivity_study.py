#!/usr/bin/env python3
"""
sensitivity_study.py — Étude de sensibilité des hyperparamètres CausalVGAE V7
================================================================================

Ce script effectue une étude de sensibilité systématique en variant UN paramètre
à la fois (les autres étant fixés à leur valeur par défaut). Pour chaque config,
on entraîne l'explainer et on collecte les courbes Fid+ vs Sparsity.

Paramètres étudiés :
  1. λ_cfx   (poids de la loss counterfactuelle)   — nécessité
  2. λ_ind   (poids de la loss d'indépendance)      — disentanglement
  3. target_spa (cible de sparsité)                  — parcimonie
  4. lr       (learning rate)                        — optimisation
  5. hidden_dim (dimension cachée du mask_net)       — capacité du réseau

Sortie : un fichier PNG par dataset×modèle avec un subplot par paramètre,
         montrant Fid+ vs Sparsity pour chaque valeur testée.

Usage :
  python sensitivity_study.py --datasets wikipedia reddit --models tgn tgat
  python sensitivity_study.py --datasets wikipedia --models tgn --quick
"""

import os, sys, argparse, time, math, inspect, random, itertools, json
from copy import deepcopy
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ─── Projet ───────────────────────────────────────────────────────────────
PROJECT_ROOT = "/content/drive/MyDrive/causalvgae/CausalVGAE_Project"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from torch_geometric.loader import TemporalDataLoader
from utils.data_loader import get_dataset, temporal_train_val_test_split
from explainer.CausalVGAE_Explainer import CausalVGAE_Explainer

# Réutiliser les fonctions utilitaires du script d'entraînement
from train_explainer import (
    set_seed, load_base_model, _get_embeddings, _fill_memory,
    compute_global_baseline, compute_metrics_with_curves
)


# ════════════════════════════════════════════════════════════════════════════
# GRILLES DE PARAMÈTRES À BALAYER
# ════════════════════════════════════════════════════════════════════════════

# Valeurs par défaut (baseline de comparaison)
DEFAULTS = {
    'lambda_cfx': 3.0,
    'lambda_ind': 0.5,
    'target_spa': 0.20,
    'lr':         5e-4,
    'hidden_dim': 128,
}

# Grilles de sensibilité — chaque paramètre est varié indépendamment
PARAM_GRIDS = {
    'lambda_cfx': [0.5, 1.0, 2.0, 3.0, 5.0, 8.0],
    'lambda_ind': [0.0, 0.1, 0.25, 0.5, 1.0, 2.0],
    'target_spa': [0.05, 0.10, 0.15, 0.20, 0.25, 0.35],
    'lr':         [1e-4, 3e-4, 5e-4, 1e-3, 2e-3],
    'hidden_dim': [64, 128, 256, 512],
}

# Grille réduite pour --quick
PARAM_GRIDS_QUICK = {
    'lambda_cfx': [1.0, 3.0, 5.0],
    'lambda_ind': [0.0, 0.5, 1.0],
    'target_spa': [0.10, 0.20, 0.30],
    'lr':         [3e-4, 5e-4, 1e-3],
    'hidden_dim': [64, 128, 256],
}

# Seuils de sparsité pour les courbes Fid+ vs Sparsity
SPA_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

# Labels et titres pour les plots
PARAM_LABELS = {
    'lambda_cfx': r'$\lambda_{cfx}$',
    'lambda_ind': r'$\lambda_{ind}$',
    'target_spa': r'$s^{*}$ (target sparsity)',
    'lr':         'Learning Rate',
    'hidden_dim': r'$d_{hidden}$',
}


# ════════════════════════════════════════════════════════════════════════════
# Entraînement paramétrique
# ════════════════════════════════════════════════════════════════════════════

def train_single_config(base_model, model_type, train_loader, val_loader,
                        test_loader, device, input_dim, all_z_train,
                        config, max_epochs=20, patience=8, seed=42):
    """
    Entraîne un explainer avec une config donnée et retourne les courbes
    Fid+ vs Sparsity sur test.

    config : dict avec clés lambda_cfx, lambda_ind, target_spa, lr, hidden_dim
    """
    set_seed(seed)

    lambda_cfx = config['lambda_cfx']
    lambda_ind = config['lambda_ind']
    target_spa = config['target_spa']
    lr         = config['lr']
    hidden_dim = config['hidden_dim']
    default_lambda_ind = 0.5  # Valeur hardcodée dans _full_loss

    # Créer l'explainer avec hidden_dim variable
    explainer = CausalVGAE_Explainer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=64
    ).to(device)
    explainer.set_baseline(all_z_train)

    optimizer = torch.optim.Adam(explainer.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-5)

    best_char = 0.0
    best_state = None
    pat_counter = 0

    for epoch in range(1, max_epochs + 1):
        tau = max(0.3, 1.0 - (epoch - 1) / max(max_epochs - 1, 1))
        explainer.train()

        if model_type == 'tgn':
            base_model.memory.reset_state()

        total_loss, n_batches = 0.0, 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                msg = batch.msg if batch.msg.size(1) > 1 else None
                z_src, z_dst = _get_embeddings(
                    base_model, batch.src, batch.dst, batch.t, msg)
                y_orig = torch.sigmoid(
                    base_model.predictor(z_src, z_dst).squeeze(-1))
                if model_type == 'tgn':
                    base_model.memory.update_state(
                        batch.src, batch.dst, batch.t, batch.msg)
                    base_model.memory.detach()

            _, _, zc, zp = explainer(
                z_src, z_dst,
                adjacency_exists=torch.ones(z_src.size(0), device=device),
                temperature=tau)

            loss, L_fid, L_cfx, L_spa, L_ind = explainer._full_loss(
                base_model, z_src, z_dst, y_orig, zc, zp,
                epoch=epoch, target_spa=target_spa, lambda_cfx=lambda_cfx)

            # Ajustement externe de lambda_ind
            if lambda_ind != default_lambda_ind:
                loss = loss + (lambda_ind - default_lambda_ind) * L_ind

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(explainer.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if n_batches == 0:
            continue

        # Validation
        if model_type == 'tgn':
            _fill_memory(base_model, model_type, train_loader, device, reset_first=True)

        val_m = compute_metrics_with_curves(
            explainer, base_model, model_type, val_loader, device,
            update_memory=True, spa_thresholds=SPA_THRESHOLDS)
        scheduler.step(val_m['CHAR'])

        if val_m['CHAR'] > best_char:
            best_char = val_m['CHAR']
            best_state = deepcopy(explainer.state_dict())
            pat_counter = 0
        else:
            pat_counter += 1
            if pat_counter >= patience:
                break

    # Charger le meilleur modèle et évaluer sur test
    if best_state is not None:
        explainer.load_state_dict(best_state)

    if model_type == 'tgn':
        _fill_memory(base_model, model_type, train_loader, device, reset_first=True)
        _fill_memory(base_model, model_type, val_loader, device, reset_first=False)

    test_m = compute_metrics_with_curves(
        explainer, base_model, model_type, test_loader, device,
        update_memory=True, spa_thresholds=SPA_THRESHOLDS)

    # Aussi récupérer les métriques val finales
    if model_type == 'tgn':
        _fill_memory(base_model, model_type, train_loader, device, reset_first=True)
    val_m_final = compute_metrics_with_curves(
        explainer, base_model, model_type, val_loader, device,
        update_memory=True, spa_thresholds=SPA_THRESHOLDS)

    return {
        'test': test_m,
        'val': val_m_final,
        'best_val_char': best_char,
    }


# ════════════════════════════════════════════════════════════════════════════
# Boucle de sensibilité
# ════════════════════════════════════════════════════════════════════════════

def run_sensitivity(model_type, dataset_name, grids, seed=42,
                    max_epochs=20, patience=8):
    """
    Pour chaque paramètre dans `grids`, on varie sa valeur (les autres = default)
    et on collecte la courbe Fid+ vs Sparsity sur test.

    Retourne : dict[param_name] → list of (param_value, fp_curve, fm_curve, metrics)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)

    print(f"\n{'━'*70}")
    print(f"  SENSITIVITY STUDY : {model_type.upper()} × {dataset_name.upper()}")
    print(f"{'━'*70}")

    # Charger les données
    result = get_dataset(root_dir=f'{PROJECT_ROOT}/data', name=dataset_name)
    data = result.data
    train_data, val_data, test_data = temporal_train_val_test_split(data)
    train_loader = TemporalDataLoader(train_data, batch_size=128, shuffle=False)
    val_loader   = TemporalDataLoader(val_data,   batch_size=128, shuffle=False)
    test_loader  = TemporalDataLoader(test_data,  batch_size=128, shuffle=False)

    # Charger le modèle de base
    base_model = load_base_model(model_type, dataset_name, device)

    # Déterminer input_dim
    with torch.no_grad():
        sb = next(iter(train_loader)).to(device)
        msg_s = sb.msg if sb.msg.size(1) > 1 else None
        zs, _ = _get_embeddings(base_model, sb.src, sb.dst, sb.t, msg_s)
        input_dim = zs.size(-1)

    # Calculer la baseline globale (une seule fois)
    print("  Calcul de la baseline globale (une seule fois)...")
    all_z_train = compute_global_baseline(base_model, model_type, train_loader, device)

    results = {}
    total_configs = sum(len(vals) for vals in grids.values())
    config_idx = 0

    for param_name, param_values in grids.items():
        print(f"\n  ── Paramètre : {param_name} ──")
        results[param_name] = []

        for pval in param_values:
            config_idx += 1
            # Construire la config : tout à default sauf le paramètre courant
            config = dict(DEFAULTS)
            config[param_name] = pval

            label = f"{param_name}={pval}"
            print(f"    [{config_idx}/{total_configs}] {label} ...", end=' ', flush=True)

            t0 = time.time()
            try:
                res = train_single_config(
                    base_model, model_type, train_loader, val_loader,
                    test_loader, device, input_dim, all_z_train,
                    config=config, max_epochs=max_epochs,
                    patience=patience, seed=seed)

                elapsed = time.time() - t0
                test_m = res['test']
                print(f"CHAR={test_m['CHAR']:.4f}  "
                      f"Fid+@0.20={test_m['Fid+']:.3f}  "
                      f"Fid-@0.20={test_m['Fid-']:.3f}  "
                      f"({elapsed:.0f}s)")

                results[param_name].append({
                    'value': pval,
                    'fp_curve': test_m['fp_curve'],
                    'fm_curve': test_m['fm_curve'],
                    'CHAR': test_m['CHAR'],
                    'Fid+': test_m['Fid+'],
                    'Fid-': test_m['Fid-'],
                    'AUFSC+': test_m['AUFSC+'],
                    'Sparsity': test_m['Sparsity'],
                    'val_CHAR': res['val']['CHAR'],
                })

            except Exception as e:
                elapsed = time.time() - t0
                print(f"FAILED ({elapsed:.0f}s) — {e}")
                import traceback; traceback.print_exc()
                results[param_name].append({
                    'value': pval,
                    'fp_curve': [0.0] * len(SPA_THRESHOLDS),
                    'fm_curve': [0.0] * len(SPA_THRESHOLDS),
                    'CHAR': 0.0, 'Fid+': 0.0, 'Fid-': 0.0,
                    'AUFSC+': 0.0, 'Sparsity': 0.0, 'val_CHAR': 0.0,
                })

    return results


# ════════════════════════════════════════════════════════════════════════════
# Visualisation — Fid+ vs Sparsity pour chaque paramètre
# ════════════════════════════════════════════════════════════════════════════

def plot_sensitivity(results, model_type, dataset_name, output_dir):
    """
    Génère un PNG avec un subplot par paramètre.
    Chaque subplot montre les courbes Fid+ vs Sparsity (seuil) pour
    chaque valeur testée du paramètre.
    """
    params = list(results.keys())
    n_params = len(params)

    # Layout : 2 colonnes si ≥ 3 paramètres, sinon 1
    if n_params <= 2:
        n_cols, n_rows = n_params, 1
    elif n_params <= 4:
        n_cols, n_rows = 2, 2
    else:
        n_cols = 3
        n_rows = math.ceil(n_params / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6.5 * n_cols, 5.0 * n_rows),
                             squeeze=False)

    # Palette de couleurs distincte
    cmap = plt.cm.get_cmap('tab10')

    for idx, param_name in enumerate(params):
        ax = axes[idx // n_cols, idx % n_cols]
        entries = results[param_name]

        for i, entry in enumerate(entries):
            val   = entry['value']
            fp    = entry['fp_curve']
            spa_x = SPA_THRESHOLDS[:len(fp)]

            color = cmap(i / max(len(entries) - 1, 1))
            is_default = (val == DEFAULTS.get(param_name))

            # Ligne plus épaisse et marker différent pour la valeur par défaut
            lw = 2.5 if is_default else 1.6
            marker = 's' if is_default else 'o'
            ls = '-' if is_default else '--'

            # Format label
            if isinstance(val, float):
                if val < 0.01:
                    lbl = f"{val:.0e}"
                elif val < 1:
                    lbl = f"{val:.2f}"
                else:
                    lbl = f"{val:.1f}"
            else:
                lbl = str(val)
            if is_default:
                lbl += " (def)"

            ax.plot(spa_x, fp, color=color, linewidth=lw, linestyle=ls,
                    marker=marker, markersize=5, label=lbl, alpha=0.9)

        ax.set_xlabel('Sparsity Threshold', fontsize=11)
        ax.set_ylabel('Fid+', fontsize=11)
        ax.set_title(f'Sensitivity to {PARAM_LABELS.get(param_name, param_name)}',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=8.5, loc='best', framealpha=0.85)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_xlim(SPA_THRESHOLDS[0] - 0.01, SPA_THRESHOLDS[-1] + 0.01)
        ax.set_ylim(-0.02, 1.05)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    # Masquer les subplots inutilisés
    for idx in range(n_params, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.suptitle(
        f'CausalVGAE Sensitivity Study — {model_type.upper()} / {dataset_name}\n'
        f'Fid+ vs Sparsity Threshold (test set)',
        fontsize=15, fontweight='bold', y=1.01)

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fname = f'sensitivity_{model_type}_{dataset_name}.png'
    fpath = os.path.join(output_dir, fname)
    fig.savefig(fpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  ✅ Figure sauvegardée : {fpath}")
    return fpath


def plot_summary_heatmap(results, model_type, dataset_name, output_dir):
    """
    Heatmap résumée : pour chaque (paramètre, valeur), affiche CHAR sur test.
    Permet de visualiser rapidement quels réglages sont optimaux.
    """
    fig, ax = plt.subplots(figsize=(12, max(4, 0.7 * sum(len(v) for v in results.values()))))

    rows = []
    row_labels = []
    for param_name, entries in results.items():
        for entry in entries:
            val = entry['value']
            is_def = (val == DEFAULTS.get(param_name))
            if isinstance(val, float) and val < 0.01:
                lbl = f"{PARAM_LABELS.get(param_name, param_name)} = {val:.0e}"
            elif isinstance(val, float):
                lbl = f"{PARAM_LABELS.get(param_name, param_name)} = {val:.2g}"
            else:
                lbl = f"{PARAM_LABELS.get(param_name, param_name)} = {val}"
            if is_def:
                lbl += " ★"
            row_labels.append(lbl)
            rows.append(entry['fp_curve'])

    data = np.array(rows)
    im = ax.imshow(data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)

    ax.set_xticks(range(len(SPA_THRESHOLDS)))
    ax.set_xticklabels([f'{s:.2f}' for s in SPA_THRESHOLDS], fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel('Sparsity Threshold', fontsize=11)
    ax.set_title(f'Fid+ Heatmap — {model_type.upper()} / {dataset_name}',
                 fontsize=13, fontweight='bold')

    # Annoter les cellules
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            color = 'white' if v > 0.6 else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=7, color=color)

    fig.colorbar(im, ax=ax, label='Fid+', shrink=0.8)
    fig.tight_layout()

    fname = f'sensitivity_heatmap_{model_type}_{dataset_name}.png'
    fpath = os.path.join(output_dir, fname)
    fig.savefig(fpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✅ Heatmap sauvegardée : {fpath}")
    return fpath


def save_results_csv(results, model_type, dataset_name, output_dir):
    """Sauvegarde les résultats dans un CSV pour analyse ultérieure."""
    rows = []
    for param_name, entries in results.items():
        for entry in entries:
            row = {
                'model': model_type,
                'dataset': dataset_name,
                'param': param_name,
                'value': entry['value'],
                'CHAR': entry['CHAR'],
                'Fid+': entry['Fid+'],
                'Fid-': entry['Fid-'],
                'AUFSC+': entry['AUFSC+'],
                'Sparsity': entry['Sparsity'],
                'val_CHAR': entry['val_CHAR'],
            }
            for i, s in enumerate(SPA_THRESHOLDS):
                if i < len(entry['fp_curve']):
                    row[f'Fid+_{int(s*100):02d}'] = entry['fp_curve'][i]
                if i < len(entry['fm_curve']):
                    row[f'Fid-_{int(s*100):02d}'] = entry['fm_curve'][i]
            rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    fpath = os.path.join(output_dir, f'sensitivity_{model_type}_{dataset_name}.csv')
    df.to_csv(fpath, index=False)
    print(f"  ✅ CSV sauvegardé : {fpath}")
    return fpath


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Étude de sensibilité CausalVGAE — Fid+ vs Sparsity')
    parser.add_argument('--datasets', nargs='+',
                        default=['wikipedia'],
                        help='Datasets à évaluer')
    parser.add_argument('--models', nargs='+',
                        default=['tgn'],
                        help='Modèles de base (tgn, tgat)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_epochs', type=int, default=20,
                        help='Nombre max d\'époques par config')
    parser.add_argument('--patience', type=int, default=8,
                        help='Early stopping patience')
    parser.add_argument('--quick', action='store_true',
                        help='Grille réduite pour tests rapides')
    parser.add_argument('--params', nargs='+',
                        default=None,
                        help='Paramètres à étudier (défaut: tous). '
                             'Ex: --params lambda_cfx target_spa')
    parser.add_argument('--output_dir', type=str,
                        default=f'{PROJECT_ROOT}/results/sensitivity',
                        help='Répertoire de sortie')
    args = parser.parse_args()

    grids = PARAM_GRIDS_QUICK if args.quick else PARAM_GRIDS

    # Filtrer les paramètres si spécifié
    if args.params:
        grids = {k: v for k, v in grids.items() if k in args.params}

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   CausalVGAE — Étude de Sensibilité des Hyperparamètres    ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Datasets  : {', '.join(args.datasets):<46s}║")
    print(f"║  Models    : {', '.join(args.models):<46s}║")
    print(f"║  Params    : {', '.join(grids.keys()):<46s}║")
    print(f"║  Configs   : {sum(len(v) for v in grids.values()):<46d}║")
    print(f"║  Seed      : {args.seed:<46d}║")
    print(f"║  Quick     : {str(args.quick):<46s}║")
    print(f"╚══════════════════════════════════════════════════════════════╝")

    all_figure_paths = []

    for mt in args.models:
        for ds in args.datasets:
            try:
                results = run_sensitivity(
                    model_type=mt,
                    dataset_name=ds,
                    grids=grids,
                    seed=args.seed,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                )

                # Sauvegarder CSV
                save_results_csv(results, mt, ds, args.output_dir)

                # Générer les figures
                fig_path = plot_sensitivity(results, mt, ds, args.output_dir)
                all_figure_paths.append(fig_path)

                hm_path = plot_summary_heatmap(results, mt, ds, args.output_dir)
                all_figure_paths.append(hm_path)

            except Exception as e:
                print(f"\n  ✗ {mt}/{ds} : {e}")
                import traceback; traceback.print_exc()

    print(f"\n{'═'*60}")
    print(f"  TERMINÉ — {len(all_figure_paths)} figures générées")
    for p in all_figure_paths:
        print(f"    → {p}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()