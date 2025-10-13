# src/infer.py
import os, sys, glob
import yaml
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms

# garante import local se rodar via "python src/infer.py"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.model_resnet50 import SkinCancerResNet50

ALL_DIR = os.path.join("data", "images", "all")

def resolve_path(p: str) -> str:
    """Se o caminho não existir, procura por stem.* recursivamente em data/images/all/**."""
    if os.path.exists(p):
        return p
    stem = os.path.splitext(os.path.basename(str(p)))[0]
    cands = glob.glob(os.path.join(ALL_DIR, "**", stem + ".*"), recursive=True)
    if cands:
        return cands[0]
    return p  # vai falhar e será contabilizado

def main():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = "reports/checkpoints/best_model.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint não encontrado em {ckpt_path}. Treine antes (python -m src.train).")

    # Preferimos usar o split processado
    test_csv = "data/processed/test_metadata.csv"
    if not os.path.exists(test_csv):
        raise FileNotFoundError("data/processed/test_metadata.csv não encontrado. Rode a preparação.")

    df = pd.read_csv(test_csv)

    # Suporta tanto 'img_path' quanto 'img_id'
    if "img_path" not in df.columns:
        if "img_id" in df.columns:
            df["img_path"] = df["img_id"].apply(lambda x: os.path.join(ALL_DIR, os.path.basename(str(x))))
        else:
            raise KeyError("Nem 'img_path' nem 'img_id' existem no CSV de teste.")

    model = SkinCancerResNet50(pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((cfg.get("img_size", 224), cfg.get("img_size", 224))),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    results = []
    not_found = 0

    with torch.no_grad():
        for _, row in df.iterrows():
            orig = str(row["img_path"])
            path = resolve_path(orig)

            if not os.path.exists(path):
                not_found += 1
                results.append({"img_path": orig, "prob_maligno": None, "pred_label": None})
                continue

            image = Image.open(path).convert("RGB")
            tensor = preprocess(image).unsqueeze(0).to(device)

            logits = model(tensor)
            prob = torch.sigmoid(logits).item()
            pred = 1 if prob >= 0.5 else 0

            results.append({"img_path": path, "prob_maligno": prob, "pred_label": pred})

    os.makedirs("reports", exist_ok=True)
    out_csv = os.path.join("reports", "predictions.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)

    print(f"✅ Inferência concluída. Salvo em {out_csv}")
    print(f"🔎 Imagens não encontradas mesmo com fallback: {not_found}")

if __name__ == "__main__":
    main()
