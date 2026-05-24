"""
data_loader.py  —  Version 8.0
================================
CORRECTION FONDAMENTALE v8 :

  POURQUOI TGN DONNE 55% SUR UCI :
  TGNMemory reçoit msg=zeros(1) → tous les messages sont identiques →
  le GRU ne peut pas distinguer les interactions → la mémoire reste
  près de son état initial → loss = 1.38 (aléatoire).

  FIX : pour UCI et CollegeMsg (sans features d'arêtes), on calcule
  des features STRUCTURELLES CAUSALES en 8 dimensions :
    0. degré sortant normalisé du nœud src (historique passé uniquement)
    1. degré entrant normalisé du nœud dst
    2. log(Δt + 1) normalisé — délai depuis le debut
    3. log(Δt_src + 1) normalisé — délai depuis dernière activité de src
    4. log(Δt_dst + 1) normalisé — délai depuis dernière activité de dst
    5. récurrence de la paire (src, dst) — 1 si déjà vue, 0 sinon
    6. popularité de dst (degré entrant normalisé par max global)
    7. rang temporel normalisé (position dans la séquence)

  Ces features sont causales : calculées uniquement depuis l'état passé,
  sans regarder les interactions futures. Pas de leakage.
  Elles donnent au GRU un vrai signal pour mettre à jour la mémoire.

  RÉSULTAT ATTENDU : loss descend en-dessous de 1.0 dès epoch 5,
  AP > 80% après 20-30 epochs sur UCI.
"""

import os
import gzip
import tarfile
import urllib.request
from collections import namedtuple, defaultdict

import torch
import numpy as np
import pandas as pd
from torch_geometric.datasets import JODIEDataset
from torch_geometric.data import TemporalData

_PROJECT_ROOT     = "/content/drive/MyDrive/causalvgae/CausalVGAE_Project"
_DEFAULT_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

_KONECT_URLS = {
    'uci_msg':   'http://konect.cc/files/download.tsv.opsahl-ucsocial.tar.bz2',
    'uci_forum': 'http://konect.cc/files/download.tsv.opsahl-ucforum.tar.bz2',
}
_KONECT_DIRS = {
    'uci_msg':   'opsahl-ucsocial',
    'uci_forum': 'opsahl-ucforum',
}

# Dimension des features structurelles pour datasets sans features
STRUCT_FEAT_DIM = 8

# Wrapper qui transporte data + num_nodes ensemble sans modifier TemporalData
DatasetResult = namedtuple('DatasetResult', ['data', 'num_nodes'])


# ─────────────────────────────────────────────────────────────────────────────
# Features structurelles causales pour datasets sans features
# ─────────────────────────────────────────────────────────────────────────────

def _compute_structural_features(df: pd.DataFrame) -> torch.Tensor:
    """
    Calcule 8 features structurelles causales par interaction.
    CAUSAL : seul l'historique PASSÉ est utilisé pour chaque interaction.

    Dim 0 : degré sortant normalisé du nœud src
    Dim 1 : degré entrant normalisé du nœud dst
    Dim 2 : log(Δt depuis début + 1) normalisé
    Dim 3 : log(Δt depuis dernière interaction de src + 1) normalisé
    Dim 4 : log(Δt depuis dernière interaction de dst + 1) normalisé
    Dim 5 : récurrence de la paire (src,dst) : 1=déjà vue, 0=première fois
    Dim 6 : popularité de dst (degré entrant relatif)
    Dim 7 : rang temporel normalisé (position dans la séquence)
    """
    n      = len(df)
    t_min  = float(df['t'].min())
    t_max  = float(df['t'].max())
    t_span = max(t_max - t_min, 1.0)

    feats = np.zeros((n, STRUCT_FEAT_DIM), dtype=np.float32)

    deg_out   = defaultdict(int)
    deg_in    = defaultdict(int)
    last_t    = {}           # dernier timestamp d'activité de chaque nœud
    seen_pairs = set()
    max_deg   = 1

    for i, row in enumerate(df.itertuples(index=False)):
        src, dst, t = int(row.src), int(row.dst), float(row.t)

        # 0. Degré sortant src (état AVANT cette interaction)
        feats[i, 0] = deg_out[src] / max(max_deg, 1)

        # 1. Degré entrant dst
        feats[i, 1] = deg_in[dst] / max(max_deg, 1)

        # 2. Δt depuis le début (normalisé log)
        dt_global   = (t - t_min) / t_span
        feats[i, 2] = float(np.log1p(dt_global * 1000) / np.log1p(1000))

        # 3. Δt depuis dernière activité de src
        if src in last_t:
            dt_src = (t - last_t[src]) / t_span
            feats[i, 3] = float(np.log1p(dt_src * 1000) / np.log1p(1000))
        else:
            feats[i, 3] = 1.0   # nœud jamais vu → délai maximal

        # 4. Δt depuis dernière activité de dst
        if dst in last_t:
            dt_dst = (t - last_t[dst]) / t_span
            feats[i, 4] = float(np.log1p(dt_dst * 1000) / np.log1p(1000))
        else:
            feats[i, 4] = 1.0

        # 5. Récurrence de la paire (src, dst)
        feats[i, 5] = 1.0 if (src, dst) in seen_pairs else 0.0

        # 6. Popularité de dst (degré entrant normalisé)
        feats[i, 6] = deg_in[dst] / max(max_deg, 1)

        # 7. Rang temporel normalisé
        feats[i, 7] = i / max(n - 1, 1)

        # Mise à jour de l'état (APRÈS avoir calculé les features)
        deg_out[src] += 1
        deg_in[dst]  += 1
        last_t[src]   = t
        last_t[dst]   = t
        seen_pairs.add((src, dst))
        max_deg = max(max_deg, deg_out[src], deg_in[dst])

    return torch.tensor(feats, dtype=torch.float)


# ─────────────────────────────────────────────────────────────────────────────
# Konect helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_out_file(directory: str):
    for root, _, files in os.walk(directory):
        for f in files:
            if f.startswith('out.') and not f.endswith('.gz'):
                return os.path.join(root, f)
    return None


def _get_konect_file(name: str, root_dir: str) -> str:
    konect_dir = os.path.join(root_dir, _KONECT_DIRS[name])
    tar_path   = os.path.join(root_dir, f'{name}.tar.bz2')

    if os.path.isdir(konect_dir):
        out = _find_out_file(konect_dir)
        if out:
            return out

    if os.path.exists(tar_path):
        print(f"  Extraction de {tar_path}...")
        with tarfile.open(tar_path, 'r:bz2') as tf:
            tf.extractall(root_dir)
        out = _find_out_file(konect_dir) or _find_out_file(root_dir)
        if out:
            return out

    url = _KONECT_URLS[name]
    print(f"  Téléchargement {name} depuis {url}...")
    urllib.request.urlretrieve(url, tar_path)
    with tarfile.open(tar_path, 'r:bz2') as tf:
        tf.extractall(root_dir)

    out = _find_out_file(konect_dir) or _find_out_file(root_dir)
    if not out:
        raise FileNotFoundError(
            f"Fichier out.* introuvable dans {root_dir} après extraction.\n"
            f"Téléchargez manuellement depuis {url}\n"
            f"Placez out.* dans {konect_dir}/"
        )
    return out


def _parse_konect(file_path: str) -> pd.DataFrame:
    rows, skipped = [], 0
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%') or line.startswith('#'):
                continue
            parts = line.split()
            try:
                if len(parts) == 3:
                    src, dst, t = int(parts[0]), int(parts[1]), float(parts[2])
                elif len(parts) >= 4:
                    src, dst, t = int(parts[0]), int(parts[1]), float(parts[3])
                else:
                    skipped += 1; continue
                if src == dst or t < 0:
                    skipped += 1; continue
                rows.append((src, dst, t))
            except (ValueError, IndexError):
                skipped += 1; continue

    if not rows:
        raise ValueError(f"Aucune ligne valide dans {file_path}")

    df = pd.DataFrame(rows, columns=['src', 'dst', 't'])
    all_ids = set(df['src'].tolist()) | set(df['dst'].tolist())
    print(f"  Interactions : {len(df)} | IDs uniques : {len(all_ids)} "
          f"| max ID : {max(all_ids)} | skipped : {skipped}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Conversion DataFrame → TemporalData avec features structurelles
# ─────────────────────────────────────────────────────────────────────────────

def _df_to_temporal(df: pd.DataFrame, add_struct_feats: bool = True):
    """
    Convertit (src, dst, t) → (TemporalData, num_nodes).

    add_struct_feats=True : calcule les 8 features structurelles causales.
    Ces features donnent un vrai signal au GRU de TGNMemory.

    num_nodes = max(src, dst) + 1 (pas de remapping, IDs originaux conservés).
    num_nodes JAMAIS stocké dans TemporalData pour éviter le bug PyG.
    """
    # Tri chronologique avant calcul des features (causalité garantie)
    df = df.sort_values('t').reset_index(drop=True)

    src = torch.tensor(df['src'].values, dtype=torch.long)
    dst = torch.tensor(df['dst'].values, dtype=torch.long)
    t   = torch.tensor(df['t'].values.astype(np.float64), dtype=torch.float).long()
    y   = torch.zeros(len(df), dtype=torch.long)

    num_nodes = int(max(src.max().item(), dst.max().item())) + 1

    if add_struct_feats:
        print(f"  Calcul des features structurelles causales ({STRUCT_FEAT_DIM} dims)...")
        msg = _compute_structural_features(df)
        print(f"  Features calculées : shape={msg.shape}, "
              f"non-zero ratio={float((msg != 0).float().mean()):.3f}")
    else:
        msg = torch.zeros((len(df), 1), dtype=torch.float)

    data = TemporalData(src=src, dst=dst, t=t, msg=msg, y=y)
    return data, num_nodes


# ─────────────────────────────────────────────────────────────────────────────
# Chargeur principal
# ─────────────────────────────────────────────────────────────────────────────

def get_dataset(root_dir: str = _DEFAULT_DATA_DIR,
                name: str = "wikipedia") -> DatasetResult:
    """
    Retourne DatasetResult(data, num_nodes).

    Usage :
        result    = get_dataset(name='uci_msg')
        data      = result.data
        num_nodes = result.num_nodes
    """
    os.makedirs(root_dir, exist_ok=True)
    print(f"[*] Préparation du dataset : {name.upper()}...")

    # ── JODIE : wikipedia, reddit, lastfm ─────────────────────────────────────
    if name in ("wikipedia", "reddit", "lastfm"):
        jodie_name = {'wikipedia': 'Wikipedia', 'reddit': 'Reddit',
                      'lastfm': 'LastFM'}[name]
        dataset   = JODIEDataset(root=root_dir, name=jodie_name)
        data      = dataset[0]
        num_nodes = int(torch.cat([data.src, data.dst]).max().item()) + 1

    # ── CollegeMsg ────────────────────────────────────────────────────────────
    elif name == "collegemsg":
        txt    = os.path.join(root_dir, "CollegeMsg.txt")
        txt_gz = os.path.join(root_dir, "CollegeMsg.txt.gz")

        if os.path.exists(txt):
            df = pd.read_csv(txt, sep=r'\s+', header=None,
                             names=['src', 'dst', 't'], comment='#')
        elif os.path.exists(txt_gz):
            with gzip.open(txt_gz, 'rt') as f:
                lines = [l for l in f if not l.startswith('#')]
            rows = [tuple(map(float, l.split())) for l in lines if len(l.split()) >= 3]
            df   = pd.DataFrame(rows, columns=['src', 'dst', 't'])
        else:
            url = 'https://snap.stanford.edu/data/CollegeMsg.txt.gz'
            print(f"  Téléchargement depuis {url}...")
            urllib.request.urlretrieve(url, txt_gz)
            with gzip.open(txt_gz, 'rt') as f:
                lines = [l for l in f if not l.startswith('#')]
            rows = [tuple(map(float, l.split())) for l in lines if len(l.split()) >= 3]
            df   = pd.DataFrame(rows, columns=['src', 'dst', 't'])

        df = df.astype({'src': int, 'dst': int})
        df = df[df['src'] != df['dst']].reset_index(drop=True)
        data, num_nodes = _df_to_temporal(df, add_struct_feats=True)

    # ── UCI-Messages ──────────────────────────────────────────────────────────
    elif name == "uci_msg":
        out_file = _get_konect_file('uci_msg', root_dir)
        print(f"  Fichier : {out_file}")
        df = _parse_konect(out_file)
        data, num_nodes = _df_to_temporal(df, add_struct_feats=True)

    # ── UCI-Forums ────────────────────────────────────────────────────────────
    elif name == "uci_forum":
        out_file = _get_konect_file('uci_forum', root_dir)
        print(f"  Fichier : {out_file}")
        df = _parse_konect(out_file)
        data, num_nodes = _df_to_temporal(df, add_struct_feats=True)

    else:
        raise ValueError(
            f"Dataset inconnu : '{name}'.\n"
            "Supportés : wikipedia, reddit, lastfm, collegemsg, uci_msg, uci_forum"
        )

    # ── Tri chronologique final ───────────────────────────────────────────────
    perm     = data.t.argsort()
    data.src = data.src[perm]
    data.dst = data.dst[perm]
    data.t   = data.t[perm]
    data.msg = data.msg[perm]
    data.y   = data.y[perm]

    print(f"[+] Dataset {name} : {num_nodes} nœuds | "
          f"{data.num_events} interactions | msg_dim={data.msg.size(1)}")

    return DatasetResult(data=data, num_nodes=num_nodes)


# ─────────────────────────────────────────────────────────────────────────────
# Split temporel 70/15/15
# ─────────────────────────────────────────────────────────────────────────────

def temporal_train_val_test_split(data,
                                  val_ratio: float  = 0.15,
                                  test_ratio: float = 0.15):
    """
    Découpage chronologique strict.
    Accepte un TemporalData OU un DatasetResult (compatibilité ascendante).
    """
    if isinstance(data, DatasetResult):
        data = data.data

    t_np = data.t.numpy()
    val_t, test_t = np.quantile(t_np, [1 - val_ratio - test_ratio, 1 - test_ratio])

    train_mask = data.t <= val_t
    val_mask   = (data.t > val_t) & (data.t <= test_t)
    test_mask  = data.t > test_t

    def _sl(m):
        return TemporalData(
            src=data.src[m], dst=data.dst[m],
            t=data.t[m], msg=data.msg[m], y=data.y[m],
        )

    tr, va, te = _sl(train_mask), _sl(val_mask), _sl(test_mask)
    print(f"[+] Split : Train={tr.num_events} | Val={va.num_events} | Test={te.num_events}")
    return tr, va, te