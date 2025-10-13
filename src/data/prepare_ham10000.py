# src/data/prepare_ham10000.py
"""
Preparação do dataset HAM10000 para classificação binária (Melanoma vs. Outras Lesões).
Gera train/val/test splits em data/processed/.
"""

import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = "data"
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
HAM10000_DIR = os.path.join("src", "data", "raw", "ham10000")
METADATA_FILE = os.path.join(HAM10000_DIR, "HAM10000_metadata.csv")
IMAGE_DIRS = [
    os.path.join(HAM10000_DIR, "HAM10000_images_part_1"),
    os.path.join(HAM10000_DIR, "HAM10000_images_part_2"),
]


def check_files():
    """Verifica se os arquivos necessários existem."""
    print("🔍 Verificando arquivos do HAM10000...")
    
    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(
            f"❌ Arquivo de metadados não encontrado em {METADATA_FILE}\n"
            f"   Baixe em: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000"
        )
    
    print(f"✅ Metadados: {METADATA_FILE}")
    for img_dir in IMAGE_DIRS:
        if os.path.exists(img_dir):
            count = len(glob.glob(os.path.join(img_dir, "*.jpg")))
            print(f"✅ {os.path.basename(img_dir)}: {count} imagens")


def build_image_index():
    """Cria índice {image_id: caminho_completo} para localização rápida."""
    print("\n🗂️  Indexando imagens...")
    index = {}
    
    for img_dir in IMAGE_DIRS:
        if not os.path.exists(img_dir):
            continue
        for img_path in glob.glob(os.path.join(img_dir, "*.jpg")):
            image_id = os.path.splitext(os.path.basename(img_path))[0]
            index[image_id] = img_path
    
    print(f"✅ {len(index)} imagens indexadas")
    return index


def create_binary_labels(df):
    """Converte diagnósticos em rótulos binários: 1=Melanoma, 0=Outras."""
    print("\n🏷️  Criando rótulos binários...")
    print(f"📊 Classes originais: {df['dx'].value_counts().to_dict()}")
    
    df['target'] = df['dx'].apply(lambda x: 1 if x == 'mel' else 0)
    
    counts = df['target'].value_counts().to_dict()
    melanoma = counts.get(1, 0)
    other = counts.get(0, 0)
    print(f"� Binário: Melanoma={melanoma}, Outras={other} (razão 1:{other/melanoma:.1f})")
    
    return df


def resolve_image_paths(df, image_index):
    """Adiciona caminhos completos das imagens ao DataFrame."""
    print("\n🔗 Resolvendo caminhos...")
    df['img_path'] = df['image_id'].apply(lambda x: image_index.get(x, None))
    
    before = len(df)
    df = df.dropna(subset=['img_path']).copy()
    if len(df) < before:
        print(f"⚠️  {before - len(df)} imagens não encontradas (removidas)")
    
    print(f"✅ {len(df)} imagens com caminho válido")
    return df


def split_dataset(df):
    """Divide em treino (70%), validação (15%) e teste (15%) com estratificação."""
    print("\n✂️  Dividindo dataset...")
    
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['target'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['target'], random_state=42)
    
    for name, split_df in [("Treino", train_df), ("Val", val_df), ("Teste", test_df)]:
        mel = (split_df['target'] == 1).sum()
        other = (split_df['target'] == 0).sum()
        print(f"   ✅ {name}: {len(split_df)} | Mel={mel}, Outras={other}")
    
    return train_df, val_df, test_df


def save_processed_data(train_df, val_df, test_df):
    """Salva CSVs processados em data/processed/."""
    print("\n💾 Salvando...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    columns = ['img_path', 'target']
    train_df[columns].to_csv(os.path.join(PROCESSED_DIR, "train_metadata.csv"), index=False)
    val_df[columns].to_csv(os.path.join(PROCESSED_DIR, "val_metadata.csv"), index=False)
    test_df[columns].to_csv(os.path.join(PROCESSED_DIR, "test_metadata.csv"), index=False)
    
    print(f"✅ Salvos em {PROCESSED_DIR}: train/val/test_metadata.csv")


def main():
    """Fluxo principal de preparação."""
    print("=" * 60)
    print("🚀 PREPARAÇÃO HAM10000 - Classificação Binária")
    print("=" * 60)
    
    check_files()
    
    print(f"\n📖 Lendo metadados...")
    df = pd.read_csv(METADATA_FILE)
    print(f"✅ {len(df)} registros")
    
    image_index = build_image_index()
    df = create_binary_labels(df)
    df = resolve_image_paths(df, image_index)
    
    train_df, val_df, test_df = split_dataset(df)
    save_processed_data(train_df, val_df, test_df)
    
    print("\n🎉 CONCLUÍDO! Execute: python src/train.py")


if __name__ == "__main__":
    main()
