"""Treinamento de ResNet50 para classificação binária de melanoma (HAM10000)."""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

from data.dataset_ham10000 import HAM10000Dataset, get_transforms
from models.model_resnet50 import SkinCancerResNet50


def load_config(config_path="configs/config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def calculate_metrics(y_true, y_pred, y_scores):
    """Calcula sensitivity, specificity, precision, F1 e AUC."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    
    try:
        auc = roc_auc_score(y_true, y_scores)
    except:
        auc = 0.0
    
    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1,
        'auc': auc,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


# ==================== TREINAMENTO ====================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Treina uma época e retorna (loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Treinando")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})
    
    return running_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device):
    """Valida e retorna (loss, metrics_dict)."""
    model.eval()
    running_loss = 0.0
    all_labels, all_preds, all_scores = [], [], []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validando"):
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()
            
            running_loss += loss.item()
            all_labels.extend(labels.cpu().numpy().flatten())
            all_preds.extend(predictions.cpu().numpy().flatten())
            all_scores.extend(probs.cpu().numpy().flatten())
    
    val_loss = running_loss / len(dataloader)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), np.array(all_scores))
    
    return val_loss, metrics


def main():
    """Função principal de treinamento."""
    print("=" * 60)
    print("🚀 TREINAMENTO - MELANOMA (HAM10000)")
    print("=" * 60)
    
    config = load_config()
    print(f"\n📋 Config: {config['dataset']} | {config['epochs']} epochs | batch={config['batch_size']}")
    
    set_seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    print(f"\n📊 Carregando dados...")
    train_dataset = HAM10000Dataset("data/processed/train_metadata.csv", 
                                     get_transforms(config['img_size'], "train"))
    val_dataset = HAM10000Dataset("data/processed/val_metadata.csv",
                                   get_transforms(config['img_size'], "val"))
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                               num_workers=config['num_workers'], 
                               pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                             num_workers=config['num_workers'],
                             pin_memory=(device.type == "cuda"))
    
    print(f"✅ Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    print(f"\n⚖️  Calculando class weights...")
    train_df = pd.read_csv("data/processed/train_metadata.csv")
    n_pos = (train_df["target"] == 1).sum()
    n_neg = (train_df["target"] == 0).sum()
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device, dtype=torch.float32)
    print(f"   Melanoma: {n_pos} | Outras: {n_neg} | pos_weight={pos_weight.item():.2f}")
    
    print(f"\n🧠 Criando modelo...")
    model = SkinCancerResNet50(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    best_val_auc = 0.0
    best_epoch = 0
    
    print(f"\n{'='*60}")
    print(f"🏋️  TREINAMENTO")
    print(f"{'='*60}\n")
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"\n📅 Epoch {epoch}/{config['epochs']}")
        print("-" * 60)
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"\n📊 Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}%")
        print(f"   Val Loss: {val_loss:.4f}")
        print(f"   Accuracy: {val_metrics['accuracy']*100:.2f}% | Sensitivity: {val_metrics['sensitivity']*100:.2f}%")
        print(f"   Specificity: {val_metrics['specificity']*100:.2f}% | AUC: {val_metrics['auc']:.4f}")
        print(f"   Confusion: TP={val_metrics['tp']} TN={val_metrics['tn']} FP={val_metrics['fp']} FN={val_metrics['fn']}")
        
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_epoch = epoch
            checkpoint_path = os.path.join(config['checkpoint_dir'], 'best_model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"\n   💾 Melhor modelo salvo! (AUC: {best_val_auc:.4f})")
    
    print(f"\n{'='*60}")
    print(f"🎉 CONCLUÍDO! Melhor: Epoch {best_epoch} | AUC {best_val_auc:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
