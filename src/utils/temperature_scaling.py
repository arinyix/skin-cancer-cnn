import torch
import torch.nn as nn
from scipy.optimize import minimize
import numpy as np

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

def calibrate_temperature(model, val_loader, device="cpu"):
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            logits_list.append(outputs.cpu())
            labels_list.append(labels)
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    temp_model = TemperatureScaler()
    optimizer = torch.optim.LBFGS([temp_model.temperature], lr=0.01, max_iter=50)
    loss_fn = nn.BCEWithLogitsLoss()

    def eval_fn():
        optimizer.zero_grad()
        loss = loss_fn(temp_model(logits), labels.float().unsqueeze(1))
        loss.backward()
        return loss

    optimizer.step(eval_fn)
    print(f"🔥 Temperatura calibrada: {temp_model.temperature.item():.3f}")
    return temp_model
