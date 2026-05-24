# CausalExplainer

Causal Explanations for Temporal Graph Neural Networks

CausalExplainer is a post-hoc, model-agnostic framework for extracting causal explanations from continuous-time dynamic graphs (CTDGs). It learns a differentiable mask in the latent space to separate causal signals from spurious noise, jointly optimising sufficiency (PS), counterfactual necessity (PN) and sparsity.

License: MIT
Python 3.9+
PyTorch 2.0+
PyTorch Geometric 2.3+

Table of Contents
1. Installation
2. Data Preparation
3. Pre-trained Models
4. Training the Base TGNN
5. Training CausalExplainer
6. Evaluating and Reproducing Results
   - Quantitative Results
   - Ablation Study
   - Sensitivity Analysis
7. Project Structure


---

1. Installation

Prerequisites:
- Python 3.9 or higher
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- CUDA (optional, for GPU acceleration)

All dependencies are listed in requirements.txt.

Clone the repository and install:

cd CausalExplainer
pip install -r requirements.txt

For development (editable mode):
pip install -e .

---

2. Data Preparation

Datasets (Wikipedia, Reddit, UCI-Messages, Collegemsg) are automatically downloaded and processed the first time you run any training script.
If you prefer to manually download raw data, place files in data/<dataset>/raw/ and run:

python -m utils.data_loader --dataset wikipedia

Replace wikipedia with reddit, uci_msg or collegemsg.

---

3. Pre-trained Models

We provide pre-trained backbones (TGN, TGAT) 

To train from scratch, follow the next sections.

---

4. Training the Base TGNN

Before training the explainer, train a TGNN (TGN or TGAT) to obtain node embeddings.

# Train TGN on Wikipedia
python Scripts/train_base_models.py --model tgn --dataset wikipedia

# Train TGAT on Reddit
python Scripts/train_base_models.py --model tgat --dataset reddit

Supported models: tgn, tgat
Supported datasets: wikipedia, reddit, uci_msg, collegemsg

Trained models are saved in checkpoints/.

---

5. Training CausalExplainer

Once the base TGNN is ready, train the explainer:

python Scripts/train_explainer.py --model tgn --dataset wikipedia --ablation full

Key arguments:
--model          : tgn or tgat (required)
--dataset        : dataset name (required)
--ablation       : full, no_ind, no_cfx, no_vae, spurious_only, random_mask (default full)
--seed           : random seed (default 42)
--target_spa_override : override target sparsity s*
--lambda_cfx_override : override counterfactual loss weight
--lambda_ind_override : override independence loss weight

Logs and metrics are saved in results/. Best model checkpoint stored in checkpoints/.

---

6. Evaluating and Reproducing Results

Quantitative Results

python Scripts/run_ablation.py --datasets wikipedia reddit uci_msg --models tgn tgat

Results (CSV files) appear in results/.

Ablation Study (Figure 3)

python Scripts/run_ablation.py --ablation --datasets wikipedia --models tgn

This runs spurious_only, no_cfx and random_mask variants.

Sensitivity Analysis (Figure 4)

python Scripts/sensitivity_study.py --datasets wikipedia --models tgn --quick

Use --quick for a reduced grid; remove for full analysis. Plots saved in results/figures/.


python Scripts/plot_explanation_subgraph.py --dataset reddit --model tgn

Output images stored in figures/subgraphs/.

---

Reproducing All Paper Results

Create a script scripts/reproduce_all.sh with the following content:

#!/bin/bash
python Scripts/train_base_models.py --model tgn --dataset wikipedia
python Scripts/train_base_models.py --model tgat --dataset reddit
python Scripts/train_explainer.py --model tgn --dataset wikipedia --ablation full
python Scripts/train_explainer.py --model tgn --dataset reddit --ablation full
python Scripts/train_explainer.py --model tgat --dataset wikipedia --ablation full
python Scripts/train_explainer.py --model tgat --dataset reddit --ablation full
python Scripts/run_ablation.py --datasets wikipedia reddit --models tgn tgat
python Scripts/sensitivity_study.py --datasets wikipedia --models tgn
python Scripts/plot_explanation_subgraph.py --dataset reddit --model tgn

Then run:
chmod +x scripts/reproduce_all.sh
bash scripts/reproduce_all.sh

All tables and figures will be saved in results/ and figures/.

---

7. Project Structure

CausalExplainer/
├── README.md
├── requirements.txt
├── setup.py
├── Scripts/
│   ├── train_base_models.py
│   ├── train_explainer.py
│   ├── sensitivity_study.py
│   ├── run_ablation.py
│   └── plot_explanation_subgraph.py
├── explainer/
│   └── CausalVGAE_Explainer.py
├── models/
│   ├── tgn_model.py
│   └── tgat.py
├── utils/
│   └── data_loader.py
├── data/
├── checkpoints/
├── results/
├── CoDy/
└── figures/

