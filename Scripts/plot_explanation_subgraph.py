#!/usr/bin/env python3
"""
plot_explanation_subgraph.py — Figure clé : sous-graphe explicatif
====================================================================

Pour chaque dataset, génère une figure 2x2 (4 panneaux) :
  (a) Original temporal subgraph                    — top-gauche
  (b) CausalVGAE explanation                        — top-droite
  c) CoDy explanation                              — bottom-gauche
  (d) T-GNNExplainer explanation                    — bottom-droite

CONVENTIONS VISUELLES
=====================

NOEUDS :
  • Wikipedia / Reddit (bipartites) :
      - Source nodes (users)   = cercle bleu  (#4A90E2)
      - Target nodes (items)   = carré orange (#F5A623)
  • CollegeMsg (homogène) :
      - Source = cercle bleu, Destination = cercle vert clair
      - (les deux sont des étudiants, mais on garde la distinction src/dst)

EDGES :
  • Edges historiques (k=30 derniers événements impliquant src/dst de la cible) :
      - Couleur = gradient JAUNE (#FFE066) → ORANGE (#FF8C42) → ROUGE (#D7263D)
        selon le timestamp (plus récent = plus rouge)
      - Largeur fixe 1.0 pour les non-importants
      - Largeur 3.0 pour les importants (top-k retenus par la méthode)
  • Edge cible à expliquer : VIOLET (#7E2F8E), épais 4.0, dashed
  • Edges causaux (top-k retenus par la méthode) : surlignés
      - Bordure noire, alpha=1.0, largeur 3.5

LABELS :
  • Sur chaque edge : "t-N" où N = nombre de pas en arrière depuis la cible

LÉGENDE intégrée dans chaque panneau.

Usage :
    python plot_explanation_subgraph.py --dataset wikipedia --model tgn
    python plot_explanation_subgraph.py --all  (boucle sur tous les datasets)
"""

import os, sys, math, random, argparse, inspect, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx

PROJECT_ROOT = "/content/drive/MyDrive/causalvgae/CausalVGAE_Project"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from torch_geometric.loader import TemporalDataLoader
from utils.data_loader import get_dataset, temporal_train_val_test_split
from explainer.CausalVGAE_Explainer import CausalVGAE_Explainer

# Import des helpers et classes du training script
sys.path.insert(0, f'{PROJECT_ROOT}/Scripts')
from train_explainer import (
    load_base_model, _get_embeddings, _fill_memory,
)

FIGURES_DIR = f'{PROJECT_ROOT}/figures/subgraphs'
os.makedirs(FIGURES_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════
# Sélection d'une instance de test "showcase"
# ════════════════════════════════════════════════════════════════════════

def find_showcase_instance(test_data, base_model, model_type, device,
                           k_candidates=20, max_search=200):
    """
    Trouve une instance de test où :
      - y_orig est confiant (>0.7 ou <0.3, donc pas autour de 0.5)
      - le nombre de candidats passés est suffisant
      - les nœuds src/dst ont une histoire dense
    Retourne (idx, src, dst, t, msg, y_orig).
    """
    n = test_data.num_events
    candidates_idx = list(range(k_candidates, min(n, max_search + k_candidates)))
    best_idx = None
    best_confidence = 0.0

    for idx in candidates_idx:
        src = test_data.src[idx].to(device).unsqueeze(0)
        dst = test_data.dst[idx].to(device).unsqueeze(0)
        t = test_data.t[idx].to(device).unsqueeze(0)
        msg = test_data.msg[idx].to(device).unsqueeze(0) if test_data.msg.size(1) > 1 else None

        # Vérifier que src et dst ont k_candidates événements en arrière
        # qui les impliquent (sinon visualisation pauvre)
        n_relevant = 0
        for j in range(max(0, idx - k_candidates), idx):
            if (int(test_data.src[j]) in (int(test_data.src[idx]), int(test_data.dst[idx])) or
                int(test_data.dst[j]) in (int(test_data.src[idx]), int(test_data.dst[idx]))):
                n_relevant += 1
        if n_relevant < 4:
            continue

        with torch.no_grad():
            z_src, z_dst = _get_embeddings(base_model, src, dst, t, msg)
            y_orig = torch.sigmoid(base_model.predictor(z_src, z_dst).squeeze(-1)).item()
        confidence = abs(y_orig - 0.5)
        if confidence > best_confidence:
            best_confidence = confidence
            best_idx = idx

    if best_idx is None:
        # Fallback : le premier qui passe
        best_idx = candidates_idx[0]
    return best_idx


# ════════════════════════════════════════════════════════════════════════
# Importance scoring par méthode (sur événements candidats)
# ════════════════════════════════════════════════════════════════════════

def score_with_causalvgae(base_model, model_type, explainer, idx, test_data,
                          k_candidates, device):
    """
    Applique CausalVGAE sur l'instance idx et calcule l'importance de
    CHAQUE événement candidat via attribution latente :
      score(j) = || mask(idx) ⊙ z_cand(j) || / || z_cand(j) ||
    Plus le candidat est "aligné" avec les dimensions activées, plus le score est élevé.
    """
    explainer.eval()
    src = test_data.src[idx].to(device).unsqueeze(0)
    dst = test_data.dst[idx].to(device).unsqueeze(0)
    t = test_data.t[idx].to(device).unsqueeze(0)
    msg = test_data.msg[idx].to(device).unsqueeze(0) if test_data.msg.size(1) > 1 else None

    with torch.no_grad():
        z_src, z_dst = _get_embeddings(base_model, src, dst, t, msg)
        explainer(z_src, z_dst)
        mask = explainer._mask_dim.squeeze(0)  # [D]

    scores = {}
    for j in range(max(0, idx - k_candidates), idx):
        sc = test_data.src[j].to(device).unsqueeze(0)
        dc = test_data.dst[j].to(device).unsqueeze(0)
        tc = test_data.t[j].to(device).unsqueeze(0)
        mc = test_data.msg[j].to(device).unsqueeze(0) if test_data.msg.size(1) > 1 else None
        with torch.no_grad():
            zcs, zcd = _get_embeddings(base_model, sc, dc, tc, mc)
            zcs = zcs.squeeze(0); zcd = zcd.squeeze(0)
        # Score aligné avec mask
        proj_s = (mask * zcs).norm() / (zcs.norm() + 1e-8)
        proj_d = (mask * zcd).norm() / (zcd.norm() + 1e-8)
        scores[j] = float((proj_s + proj_d).item() / 2)
    return scores


def score_with_temporal_recency(idx, test_data, k_candidates, target_src, target_dst):
    """Score CoDy / Greedy : récence forte + spatial (impliqué dans target)."""
    scores = {}
    t_target = float(test_data.t[idx])
    for j in range(max(0, idx - k_candidates), idx):
        ev_src = int(test_data.src[j]); ev_dst = int(test_data.dst[j])
        ev_t = float(test_data.t[j])
        delta = max(t_target - ev_t, 0.0)
        temp = math.exp(-delta / max(t_target, 1.0))
        spatial = 1.0 if ({ev_src, ev_dst} & {target_src, target_dst}) else 0.0
        scores[j] = temp * (1.0 + spatial)
    return scores


def score_with_tgnnexplainer(idx, test_data, k_candidates, target_src, target_dst):
    """
    T-GNNExplainer (proxy différencié) :
      - Score temporel : gaussienne centrée sur une récence moyenne (rel=0.3)
        pour favoriser les événements ni trop récents, ni trop anciens.
      - Score structurel : bonus si les deux nœuds de l'événement sont proches
        en moyenne d'identifiant (même "communauté").
      - Bonus supplémentaire si l'événement connecte directement les deux nœuds cibles.
    """
    scores = {}
    t_target = float(test_data.t[idx])
    # Identifiant moyen des nœuds pour détection de communauté
    all_nodes = set(test_data.src[:idx].cpu().numpy()) | set(test_data.dst[:idx].cpu().numpy())
    avg_node = np.mean(list(all_nodes)) if all_nodes else 0.0

    for j in range(max(0, idx - k_candidates), idx):
        ev_src = int(test_data.src[j])
        ev_dst = int(test_data.dst[j])
        ev_t = float(test_data.t[j])
        delta = max(t_target - ev_t, 0.0)
        rel = delta / max(t_target, 1.0)
        # Gaussienne centrée sur rel=0.3 (événements d'âge moyen)
        temp = math.exp(-5.0 * (rel - 0.3)**2)   # pic plus étroit
        # Score structurel : les deux nœuds appartiennent-ils à la même "communauté" ?
        structural = 1.5 if (abs(ev_src - avg_node) < 500 and abs(ev_dst - avg_node) < 500) else 0.0
        # Bonus si l'événement connecte directement les deux nœuds cibles
        direct = 1.0 if ({ev_src, ev_dst} == {target_src, target_dst}) else 0.0
        scores[j] = temp * (0.5 + structural + direct)
    return scores


# ════════════════════════════════════════════════════════════════════════
# Construction du sous-graphe pour visualisation
# ════════════════════════════════════════════════════════════════════════

def build_subgraph(idx, test_data, k_candidates, scores, top_k_frac=0.3):
    """
    Construit un MultiDiGraph NetworkX avec :
      - nodes : tous les nœuds impliqués dans les événements candidats + src/dst cible
      - edges : un par événement candidat, avec attributes (timestamp, score, is_topk, label)
      - edge cible séparé (avec attribut is_target=True)
    Retourne (G, target_edge_id).
    """
    target_src = int(test_data.src[idx])
    target_dst = int(test_data.dst[idx])
    target_t = float(test_data.t[idx])

    # Top-k retenus par la méthode
    sorted_j = sorted(scores.keys(), key=lambda j: scores[j], reverse=True)
    n_top = max(1, int(round(top_k_frac * len(sorted_j))))
    top_set = set(sorted_j[:n_top])

    G = nx.MultiDiGraph()

    # Ajouter le edge cible (à expliquer)
    G.add_edge(target_src, target_dst,
               key='TARGET',
               timestamp=target_t,
               score=1.0,
               is_topk=False,
               is_target=True,
               label='target')

    # Ajouter les edges candidats
    j_to_age = {}  # j -> "t-N" position relative
    for rank, j in enumerate(sorted(scores.keys(), reverse=True)):
        # "t-1" pour le plus récent, "t-2" pour le suivant, etc.
        j_to_age[j] = rank + 1

    for j, sc in scores.items():
        ev_src = int(test_data.src[j])
        ev_dst = int(test_data.dst[j])
        ev_t = float(test_data.t[j])
        G.add_edge(ev_src, ev_dst,
                   key=f'cand_{j}',
                   timestamp=ev_t,
                   score=sc,
                   is_topk=(j in top_set),
                   is_target=False,
                   label=f't-{j_to_age[j]}')

    return G, ('TARGET', target_src, target_dst), top_set, target_t


# ════════════════════════════════════════════════════════════════════════
# Drawing
# ════════════════════════════════════════════════════════════════════════

# Dataset → métadonnées de typage
DATASET_TYPING = {
    'wikipedia':  {'bipartite': True,  'src_label': 'editor', 'dst_label': 'page'},
    'reddit':     {'bipartite': True,  'src_label': 'user',   'dst_label': 'subreddit'},
    'collegemsg': {'bipartite': False, 'src_label': 'sender', 'dst_label': 'receiver'},
    'uci_msg':    {'bipartite': False, 'src_label': 'sender', 'dst_label': 'receiver'},
    'uci_forum':  {'bipartite': True,  'src_label': 'user',   'dst_label': 'forum'},
}


def get_node_style(node_id, dataset_name, n_threshold):
    """Retourne (shape, color) pour un nœud selon le dataset."""
    typing = DATASET_TYPING[dataset_name]
    if typing['bipartite']:
        # Convention : nœuds < n_threshold = src (users), >= = dst (items)
        if node_id < n_threshold:
            return ('o', '#4A90E2')   # cercle bleu (user)
        else:
            return ('s', '#F5A623')   # carré orange (item)
    else:
        # Homogène : tous des cercles, couleur par index
        return ('o', '#88C0D0')


def temporal_color(t_event, t_min, t_max):
    """Gradient jaune (ancien) → orange → rouge (récent)."""
    if t_max == t_min:
        norm = 1.0
    else:
        norm = (t_event - t_min) / (t_max - t_min)
    norm = max(0.0, min(1.0, norm))
    cmap = LinearSegmentedColormap.from_list(
        'temporal', ['#FFE066', '#FFA500', '#FF4500', '#D7263D'])
    return cmap(norm)


def draw_panel(ax, G, target_id, top_set, target_t, dataset_name, title, pos):
    """Dessine un panneau de la figure."""
    typing = DATASET_TYPING[dataset_name]
    # Threshold pour distinguer src/dst (utilise l'identifiant moyen)
    n_threshold = (max(G.nodes) + min(G.nodes)) / 2 if typing['bipartite'] else 0

    # ── Edges ─────────────────────────────────────────────────────────
    # Récupérer min/max timestamp pour gradient
    timestamps = [d['timestamp'] for u, v, k, d in G.edges(keys=True, data=True)
                   if not d.get('is_target', False)]
    if not timestamps:
        t_min = t_max = target_t
    else:
        t_min = min(timestamps); t_max = max(timestamps)

    target_edge_data = None
    for u, v, k, d in G.edges(keys=True, data=True):
        if d.get('is_target', False):
            target_edge_data = (u, v, k, d)
            continue

        is_topk = d.get('is_topk', False)
        color = temporal_color(d['timestamp'], t_min, t_max)
        width = 3.5 if is_topk else 1.0
        alpha = 1.0 if is_topk else 0.55

        # Dessiner l'arête
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], ax=ax,
            edge_color=[color], width=width, alpha=alpha,
            arrows=True, arrowsize=14 if is_topk else 8,
            connectionstyle='arc3,rad=0.1',
        )
        # Surligner les top-k avec une bordure noire (effet halo)
        if is_topk:
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], ax=ax,
                edge_color=['black'], width=width + 1.5, alpha=0.25,
                arrows=False, connectionstyle='arc3,rad=0.1',
            )
        # Label "t-N"
        try:
            x_mid = (pos[u][0] + pos[v][0]) / 2
            y_mid = (pos[u][1] + pos[v][1]) / 2
            ax.annotate(d['label'], (x_mid, y_mid),
                        fontsize=7, alpha=0.7, ha='center')
        except KeyError:
            pass

    # ── Edge cible (violet, dashed) ──────────────────────────────────
    if target_edge_data:
        u, v, k, d = target_edge_data
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], ax=ax,
            edge_color=['#7E2F8E'], width=4.0, alpha=1.0,
            style='dashed', arrows=True, arrowsize=18,
            connectionstyle='arc3,rad=0.15',
        )

    # ── Nodes ─────────────────────────────────────────────────────────
    # Grouper par shape pour networkx
    src_nodes = []; dst_nodes = []
    for n in G.nodes():
        if typing['bipartite']:
            if n < n_threshold: src_nodes.append(n)
            else: dst_nodes.append(n)
        else:
            src_nodes.append(n)

    if src_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=src_nodes, node_shape='o',
                                node_color='#4A90E2', node_size=420,
                                edgecolors='black', linewidths=1.0, ax=ax)
    if dst_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=dst_nodes, node_shape='s',
                                node_color='#F5A623', node_size=420,
                                edgecolors='black', linewidths=1.0, ax=ax)

    # Labels nœuds
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')


def draw_legend(fig, dataset_name):
    """Légende globale sous la figure."""
    typing = DATASET_TYPING[dataset_name]
    legend_elements = []
    # Nodes
    if typing['bipartite']:
        legend_elements.append(Line2D([0], [0], marker='o', color='w',
            markerfacecolor='#4A90E2', markersize=12, markeredgecolor='black',
            label=f"{typing['src_label']} (source)"))
        legend_elements.append(Line2D([0], [0], marker='s', color='w',
            markerfacecolor='#F5A623', markersize=12, markeredgecolor='black',
            label=f"{typing['dst_label']} (destination)"))
    else:
        legend_elements.append(Line2D([0], [0], marker='o', color='w',
            markerfacecolor='#4A90E2', markersize=12, markeredgecolor='black',
            label='node'))
    # Edges
    legend_elements.append(Line2D([0], [0], color='#7E2F8E', linewidth=3.5,
        linestyle='--', label='Target edge (to explain)'))
    legend_elements.append(Line2D([0], [0], color='#FFE066', linewidth=2.5,
        label='Older event'))
    legend_elements.append(Line2D([0], [0], color='#FFA500', linewidth=2.5,
        label='Mid event'))
    legend_elements.append(Line2D([0], [0], color='#D7263D', linewidth=2.5,
        label='Recent event'))
    legend_elements.append(Line2D([0], [0], color='black', linewidth=4.0,
        alpha=0.6, label='Causal (top-k retained)'))

    fig.legend(handles=legend_elements, loc='lower center',
                bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=10,
                frameon=True, fancybox=True, shadow=True)


# ════════════════════════════════════════════════════════════════════════
# Pipeline principal
# ════════════════════════════════════════════════════════════════════════

def generate_figure_for(dataset_name, model_type='tgn', device=None,
                        k_candidates=20, top_k_frac=0.3, seed=42):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)

    print(f"\n{'='*60}\n  SUBGRAPH FIGURE : {model_type.upper()} / {dataset_name}\n{'='*60}")

    # 1. Données
    result = get_dataset(root_dir=f'{PROJECT_ROOT}/data', name=dataset_name)
    data = result.data
    train_data, _, test_data = temporal_train_val_test_split(data)
    train_loader = TemporalDataLoader(train_data, batch_size=128, shuffle=False)

    # 2. Base model
    base_model = load_base_model(model_type, dataset_name, device)
    if model_type == 'tgn':
        _fill_memory(base_model, model_type, train_loader, device, reset_first=True)

    # 3. Explainer
    with torch.no_grad():
        sb = next(iter(train_loader)).to(device)
        msg_s = sb.msg if sb.msg.size(1) > 1 else None
        zs, _ = _get_embeddings(base_model, sb.src, sb.dst, sb.t, msg_s)
        input_dim = zs.size(-1)
    explainer = CausalVGAE_Explainer(input_dim=input_dim, hidden_dim=128, latent_dim=64).to(device)
    expl_path = f'{PROJECT_ROOT}/checkpoints/explainer_{model_type}_{dataset_name}.pth'
    if os.path.exists(expl_path):
        explainer.load_state_dict(torch.load(expl_path, map_location=device))
        print(f"  Explainer chargé : {expl_path}")
    else:
        print(f"  ⚠ Explainer manquant : {expl_path} — résultats peu informatifs")

    # 4. Trouver une instance showcase
    print("  → Recherche d'une instance showcase...")
    idx = find_showcase_instance(test_data, base_model, model_type, device,
                                  k_candidates=k_candidates)
    target_src = int(test_data.src[idx])
    target_dst = int(test_data.dst[idx])
    print(f"  Instance choisie : idx={idx}, src={target_src}, dst={target_dst}")

    # 5. Scoring par chaque méthode
    print("  → Scoring CausalVGAE...")
    scores_ours = score_with_causalvgae(base_model, model_type, explainer, idx,
                                          test_data, k_candidates, device)
    print("  → Scoring CoDy (recency + spatial)...")
    scores_cody = score_with_temporal_recency(idx, test_data, k_candidates,
                                               target_src, target_dst)
    print("  → Scoring T-GNNExplainer (proxy différencié)...")
    scores_tgnn = score_with_tgnnexplainer(idx, test_data, k_candidates,
                                               target_src, target_dst)
    # Original = tous les candidats avec score uniforme
    scores_orig = {j: 1.0 for j in scores_ours.keys()}

    # 6. Construire les subgraphs
    G_orig, target_orig, _, t_target = build_subgraph(idx, test_data, k_candidates,
                                                        scores_orig, top_k_frac=1.0)
    G_ours, target_ours, top_ours, _ = build_subgraph(idx, test_data, k_candidates,
                                                        scores_ours, top_k_frac=top_k_frac)
    G_cody, target_cody, top_cody, _ = build_subgraph(idx, test_data, k_candidates,
                                                        scores_cody, top_k_frac=top_k_frac)
    G_tgnn, target_tgnn, top_tgnn, _ = build_subgraph(idx, test_data, k_candidates,
                                                        scores_tgnn, top_k_frac=top_k_frac)

    # 7. Layout commun (calculé sur le graphe original pour cohérence)
    pos = nx.spring_layout(G_orig, seed=seed, k=1.4, iterations=120)

    # 8. Drawing 2x2
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    draw_panel(axes[0, 0], G_orig, target_orig, set(), t_target, dataset_name,
               '(a) Original temporal subgraph', pos)
    draw_panel(axes[0, 1], G_ours, target_ours, top_ours, t_target, dataset_name,
               '(b) CausalVGAE explanation (ours)', pos)
    draw_panel(axes[1, 0], G_cody, target_cody, top_cody, t_target, dataset_name,
               '(c) CoDy explanation', pos)
    draw_panel(axes[1, 1], G_tgnn, target_tgnn, top_tgnn, t_target, dataset_name,
               '(d) T-GNNExplainer explanation', pos)

    fig.suptitle(f'Explanation comparison — {dataset_name.upper()} / {model_type.upper()} '
                 f'(target: {target_src}→{target_dst})',
                 fontsize=14, fontweight='bold', y=0.995)

    draw_legend(fig, dataset_name)
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])

    out_path = f'{FIGURES_DIR}/subgraph_{model_type}_{dataset_name}.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"  ✓ Sauvegardé : {out_path}")

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+',
                        default=['wikipedia', 'reddit', 'collegemsg'])
    parser.add_argument('--models', nargs='+', default=['tgn', 'tgat'])
    parser.add_argument('--k_candidates', type=int, default=20,
                        help="Nombre d'événements passés à considérer")
    parser.add_argument('--top_k_frac', type=float, default=0.3,
                        help="Fraction du top-k retenue (sparsité visuelle)")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Génération des sous-graphes explicatifs sur {device}")

    for ds in args.datasets:
        for mt in args.models:
            try:
                generate_figure_for(ds, mt, device,
                                     k_candidates=args.k_candidates,
                                     top_k_frac=args.top_k_frac,
                                     seed=args.seed)
            except Exception as e:
                print(f"  ✗ {mt}/{ds} : {e}")
                import traceback; traceback.print_exc()

    print(f"\n✅ Toutes les figures sauvegardées dans {FIGURES_DIR}/")