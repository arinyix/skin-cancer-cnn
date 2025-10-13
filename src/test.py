"""Script de teste no conjunto de teste do HAM10000."""

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, f1_score, confusion_matrix, brier_score_loss
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset_ham10000 import HAM10000Dataset, get_transforms
from src.models.model_resnet50 import SkinCancerResNet50


def main():
    print("=" * 60)
    print("🧪 TESTE - Avaliação no conjunto de teste")
    print("=" * 60)
    
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    
    # Carrega dataset de teste
    print("\n📊 Carregando dados de teste...")
    test_dataset = HAM10000Dataset(
        csv_path="data/processed/test_metadata.csv",
        transform=get_transforms(img_size=cfg['img_size'], phase="val")
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg.get('num_workers', 0)
    )
    
    print(f"✅ Teste: {len(test_dataset)} imagens ({len(test_loader)} batches)")
    
    # Carrega modelo
    ckpt_path = "reports/checkpoints/best_model.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"❌ Checkpoint não encontrado: {ckpt_path}\n   Execute o treinamento primeiro!")
    
    print(f"\n🧠 Carregando modelo de {ckpt_path}...")
    model = SkinCancerResNet50(pretrained=False)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("✅ Modelo carregado!")
    
    # Inferência
    print("\n🔬 Executando inferência...")
    probs, labels = [], []
    
    with torch.no_grad():
        for imgs, labs in tqdm(test_loader, desc="Testando"):
            imgs = imgs.to(device)
            labs = labs.float().to(device)
            
            logits = model(imgs)
            p = torch.sigmoid(logits).cpu().numpy().ravel()
            
            probs.extend(p)
            labels.extend(labs.cpu().numpy().ravel())
    
    probs = np.array(probs)
    labels = np.array(labels).astype(int)
    
    # Calcula métricas
    print("\n" + "=" * 60)
    print("📊 RESULTADOS - CONJUNTO DE TESTE")
    print("=" * 60)
    
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    ap = average_precision_score(labels, probs)
    
    preds_bin = (probs >= 0.5).astype(int)
    f1 = f1_score(labels, preds_bin)
    cm = confusion_matrix(labels, preds_bin)
    
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    brier = brier_score_loss(labels, probs)
    
    print(f"\n🎯 Métricas Principais:")
    print(f"   Accuracy:     {accuracy*100:.2f}%")
    print(f"   ROC-AUC:      {auc:.4f}")
    print(f"   PR-AUC:       {ap:.4f}")
    print(f"   F1-Score:     {f1:.4f}")
    print(f"   Brier Score:  {brier:.4f}")
    
    print(f"\n📈 Métricas Clínicas:")
    print(f"   Sensitivity (Recall): {sens*100:.2f}%  ← Detectou {tp}/{tp+fn} melanomas")
    print(f"   Specificity:          {spec*100:.2f}%  ← Classificou {tn}/{tn+fp} benignas corretamente")
    print(f"   Precision:            {precision*100:.2f}%  ← {tp}/{tp+fp} predições de melanoma estavam certas")
    
    print(f"\n🔢 Matriz de Confusão:")
    print(f"                 Predito")
    print(f"                 0      1")
    print(f"   Real 0 (Neg)  {tn:<6} {fp:<6}  (TN={tn}, FP={fp})")
    print(f"   Real 1 (Pos)  {fn:<6} {tp:<6}  (FN={fn}, TP={tp})")
    
    # Salva curvas
    fig_dir = "reports/figures"
    os.makedirs(fig_dir, exist_ok=True)
    
    print(f"\n📊 Gerando visualizações...")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curve - Conjunto de Teste', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(fig_dir, "test_roc_curve.png")
    plt.savefig(roc_path, dpi=150)
    print(f"   ✅ ROC Curve salva: {roc_path}")
    plt.close()
    
    # Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(labels, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(rec, prec, linewidth=2, label=f'PR (AUC={ap:.3f})')
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve - Conjunto de Teste', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    pr_path = os.path.join(fig_dir, "test_pr_curve.png")
    plt.savefig(pr_path, dpi=150)
    print(f"   ✅ PR Curve salva: {pr_path}")
    plt.close()
    
    # Confusion Matrix Heatmap
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.title('Matriz de Confusão - Conjunto de Teste', fontsize=14, fontweight='bold')
    plt.ylabel('Real', fontsize=12)
    plt.xlabel('Predito', fontsize=12)
    plt.xticks([0, 1], ['Benigna (0)', 'Melanoma (1)'])
    plt.yticks([0, 1], ['Benigna (0)', 'Melanoma (1)'])
    
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', 
                    fontsize=20, fontweight='bold', color='white' if cm[i, j] > cm.max()/2 else 'black')
    
    plt.tight_layout()
    cm_path = os.path.join(fig_dir, "test_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"   ✅ Confusion Matrix salva: {cm_path}")
    plt.close()
    
    print("\n" + "=" * 60)
    print("🎉 TESTE CONCLUÍDO!")
    print("=" * 60)
    print(f"\n📁 Visualizações salvas em: {fig_dir}/")
    print(f"   - test_roc_curve.png")
    print(f"   - test_pr_curve.png")
    print(f"   - test_confusion_matrix.png")


if __name__ == "__main__":
    main()

