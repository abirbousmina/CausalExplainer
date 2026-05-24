import sys
import os
import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import TemporalDataLoader

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.data_loader import get_dataset, temporal_train_val_test_split
from models.tgn_model import RobustTGN
from models.tgat import RobustTGAT

def train_one_epoch(model, loader, optimizer, criterion, device, num_nodes, model_type):
    model.train()
    total_loss = 0
    
    if model_type == 'tgn':
        model.memory.reset_state()

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        neg_dst = torch.randint(0, num_nodes, (batch.num_events,), dtype=torch.long, device=device)

        if model_type == 'tgn':
            z_src, z_dst = model(batch.src, batch.dst)
            _, z_neg_dst = model(batch.src, neg_dst)
        elif model_type == 'tgat':
            z_src, z_dst = model(batch.src, batch.dst, batch.t)
            _, z_neg_dst = model(batch.src, neg_dst, batch.t)
        
        pos_out = model.predictor(z_src, z_dst)
        neg_out = model.predictor(z_src, z_neg_dst)

        loss = criterion(pos_out, torch.ones_like(pos_out)) + \
               criterion(neg_out, torch.zeros_like(neg_out))

        loss.backward()
        optimizer.step()
        
        if model_type == 'tgn':
            model.update_memory(batch.src, batch.dst, batch.t, batch.msg)
            model.memory.detach()

        total_loss += float(loss) * batch.num_events

    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_model(model, loader, device, num_nodes, model_type):
    model.eval()
    y_pred, y_true = [], []

    for batch in loader:
        batch = batch.to(device)
        neg_dst = torch.randint(0, num_nodes, (batch.num_events,), dtype=torch.long, device=device)

        if model_type == 'tgn':
            z_src, z_dst = model(batch.src, batch.dst)
            _, z_neg_dst = model(batch.src, neg_dst)
            pos_out = model.predictor(z_src, z_dst)
            neg_out = model.predictor(z_src, z_neg_dst)
            model.update_memory(batch.src, batch.dst, batch.t, batch.msg)
        elif model_type == 'tgat':
            z_src, z_dst = model(batch.src, batch.dst, batch.t)
            _, z_neg_dst = model(batch.src, neg_dst, batch.t)
            pos_out = model.predictor(z_src, z_dst)
            neg_out = model.predictor(z_src, z_neg_dst)
        
        y_pred.append(pos_out.cpu())
        y_pred.append(neg_out.cpu())
        y_true.append(torch.ones_like(pos_out).cpu())
        y_true.append(torch.zeros_like(neg_out).cpu())

    y_pred = torch.cat(y_pred, dim=0).numpy()
    y_true = torch.cat(y_true, dim=0).numpy()
    
    return average_precision_score(y_true, y_pred), roc_auc_score(y_true, y_pred)

def run_experiment(model_type, dataset_name, device):
    print(f"\n{'='*50}")
    print(f"🚀 LANCEMENT EXPÉRIENCE : Modèle [{model_type.upper()}] sur Dataset [{dataset_name.upper()}]")
    print(f"{'='*50}")

    # 1. Chargement des données
    data = get_dataset(name=dataset_name)
    train_data, val_data, test_data = temporal_train_val_test_split(data)
    num_nodes = data.num_nodes

    train_loader = TemporalDataLoader(train_data, batch_size=200)
    val_loader = TemporalDataLoader(val_data, batch_size=200)
    test_loader = TemporalDataLoader(test_data, batch_size=200)

    # 2. Initialisation du modèle choisi
    if model_type == 'tgn':
        model = RobustTGN(num_nodes=num_nodes, node_features_dim=128, edge_features_dim=data.msg.size(1), memory_dim=100, time_dim=100).to(device)
    elif model_type == 'tgat':
        model = RobustTGAT(node_features_dim=128, time_dim=100, num_nodes=num_nodes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.BCELoss()

    # 3. Entraînement
    epochs = 50 # Réduit à 15 pour gagner du temps lors des tests multiples
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device, num_nodes, model_type)
        val_ap, val_auc = eval_model(model, val_loader, device, num_nodes, model_type)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val AP: {val_ap:.4f}, Val AUC: {val_auc:.4f}')

    # 4. Évaluation et Sauvegarde
    test_ap, test_auc = eval_model(model, test_loader, device, num_nodes, model_type)
    print(f'\n[*] RÉSULTAT FINAL {model_type.upper()} + {dataset_name.upper()} | AP: {test_ap:.4f}, AUC: {test_auc:.4f}')
    
    os.makedirs('checkpoints', exist_ok=True)
    save_path = f'checkpoints/{model_type}_{dataset_name}_best.pth'
    torch.save(model.state_dict(), save_path)
    print(f"[+] Modèle sauvegardé : {save_path}\n")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Exécution globale sur : {device}")

    # LA GRANDE MATRICE D'EXPÉRIENCES (Décommentez ce que vous voulez lancer)
    datasets = ['wikipedia', 'reddit', 'collegemsg'] 
    models = ['tgn', 'tgat']

    for dataset in datasets:
        for model in models:
            # Cette boucle va lancer TGN-Wiki, puis TGAT-Wiki, puis TGN-Reddit, puis TGAT-Reddit.
            run_experiment(model_type=model, dataset_name=dataset, device=device)

if __name__ == "__main__":
    main()