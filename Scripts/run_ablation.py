#!/usr/bin/env python3
"""
run_ablation_final.py
"""

import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = "/content/drive/MyDrive/causalvgae/CausalVGAE_Project"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train_explainer import run_training

def main():
    dataset = 'wikipedia'
    model = 'tgn'
    seed = 2024
    num_epochs = 10 

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   LAUNCHING CAUSAL EXPLAINER ABLATION STUDY                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print("\nNote: FULL model is trained first to provide the baseline weights")
    print("for Random Mask and Spurious Only evaluations.")

    results = {}

    # =====================================================================
    # 1. FULL Model (Must be run first to generate the checkpoint)
    # =====================================================================
    print("\n" + "━"*60)
    print(f"  [1/4] Training: FULL CAUSAL MODEL (Epochs: {num_epochs})")
    print("━"*60)
    test_full, _ = run_training(model_type=model, dataset_name=dataset, ablation='full', seed=seed, epochs=num_epochs)
    results['full'] = test_full

    # =====================================================================
    # 2. RANDOM MASK (Sanity Check)
    # =====================================================================
    print("\n" + "━"*60)
    print(f"  [2/4] Evaluation: RANDOM MASK (Sanity Check)")
    print("━"*60)
    test_random, _ = run_training(model_type=model, dataset_name=dataset, ablation='random_mask', seed=seed, epochs=0)
    results['random_mask'] = test_random

    # =====================================================================
    # 3. SPURIOUS ONLY (Inverted Mask)
    # =====================================================================
    print("\n" + "━"*60)
    print(f"  [3/4] Evaluation: SPURIOUS ONLY (Inverted Mask)")
    print("━"*60)
    test_spurious, _ = run_training(model_type=model, dataset_name=dataset, ablation='spurious_only', seed=seed, epochs=0)
    results['spurious_only'] = test_spurious

    # =====================================================================
    # 4. NO CFX (No Counterfactual)
    # =====================================================================
    print("\n" + "━"*60)
    print(f"  [4/4] Training: NO COUNTERFACTUAL (Epochs: {num_epochs})")
    print("━"*60)
    test_no_cfx, _ = run_training(model_type=model, dataset_name=dataset, ablation='no_cfx', seed=seed, epochs=num_epochs, lambda_cfx_override=0.0)
    results['no_cfx'] = test_no_cfx

    # =====================================================================
    # PLOTTING THE ABLATION FIGURE
    # =====================================================================
    print("\n📈 Generating the final publication-ready plot...")
    
    plt.figure(figsize=(10, 6))
    sns.set_style('whitegrid')

    spa_vals = [0, 5, 10, 15, 20, 25, 30]
    x_vals = [s / 100 for s in spa_vals]

    curve_full = [results['full'][f'Fid+_{s}'] for s in spa_vals]
    curve_random = [results['random_mask'][f'Fid+_{s}'] for s in spa_vals]
    curve_spurious = [results['spurious_only'][f'Fid+_{s}'] for s in spa_vals]
    curve_no_cfx = [results['no_cfx'][f'Fid+_{s}'] for s in spa_vals]

    # Plotting in visually distinct styles
    plt.plot(x_vals, curve_full, color='blue', marker='s', linestyle='-', linewidth=3, markersize=9, label='Full Causal Model')
    plt.plot(x_vals, curve_no_cfx, color='red', marker='x', linestyle='--', linewidth=2.5, markersize=8, label='No Counterfactual (No CFX)')
    plt.plot(x_vals, curve_spurious, color='black', marker='o', linestyle='-', linewidth=2.5, markersize=8, label='Spurious Only (Inverted Mask)')
    plt.plot(x_vals, curve_random, color='gray', marker='D', linestyle=':', linewidth=2.5, markersize=8, label='Random Mask (Sanity Check)')

    plt.xlabel('Sparsity', fontsize=12, fontweight='bold')
    plt.ylabel('Fidelity+ (Sufficiency)', fontsize=12, fontweight='bold')
    plt.title(f"Ablation Study: Impact on Explanatory Sufficiency\nModel: {model.upper()} - Dataset: {dataset.capitalize()}", fontsize=14, fontweight='bold')
    
    plt.ylim(0, 1.05)
    plt.xticks(x_vals)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2, fontsize=11)

    save_dir = os.path.join(PROJECT_ROOT, 'results', 'figures')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'ablation_final_english_{model}_{dataset}.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Study complete! Publication-ready plot saved to: \n ➡️ {save_path}\n")

if __name__ == "__main__":
    main()