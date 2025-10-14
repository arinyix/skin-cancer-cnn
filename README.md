# 🔬 Detector de Melanoma com CNN

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Um aplicativo de **Deep Learning** para detecção de melanoma em imagens dermatoscópicas, utilizando **ResNet50** e interface interativa com **Streamlit**. O modelo realiza classificação binária (Melanoma vs. Outras Lesões) com base no dataset **HAM10000**, alcançando **AUC-ROC de 0.86** em testes iniciais.

---

## 📋 Descrição

Este projeto implementa um sistema de auxílio ao diagnóstico de lesões de pele, utilizando técnicas de **Transfer Learning** com a arquitetura **ResNet50** pré-treinada no ImageNet. O modelo foi treinado para classificar imagens dermatoscópicas em duas categorias:

- **Classe 0:** Outras Lesões (nevus melanocítico, carcinoma basocelular, queratose actínica, etc.)
- **Classe 1:** **MELANOMA** (tipo mais perigoso de câncer de pele)

O sistema oferece uma interface web intuitiva desenvolvida em **Streamlit**, permitindo upload de imagens e análise em tempo real com ajuste de threshold de decisão.

> **⚠️ AVISO IMPORTANTE:** Este é um projeto educacional e **não substitui** avaliação médica profissional. Sempre consulte um dermatologista para diagnóstico definitivo.

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.13** - Linguagem de programação
- **PyTorch 2.8** - Framework de Deep Learning
- **torchvision** - Modelos pré-treinados e transformações de imagem
- **Streamlit** - Interface web interativa
- **Albumentations** - Data augmentation avançado
- **Pandas** - Manipulação de dados
- **NumPy** - Operações numéricas
- **Scikit-learn** - Métricas de avaliação
- **Matplotlib** - Visualizações (ROC curve, confusion matrix)
- **OpenCV** - Processamento de imagem
- **PIL (Pillow)** - Leitura de imagens

---

## 📁 Estrutura do Projeto

```
skin-cancer-cnn/
├── app.py                          # Aplicativo Streamlit
├── configs/
│   └── config.yaml                 # Configurações de treinamento
├── data/
│   ├── processed/                  # CSVs com splits train/val/test
│   └── raw/
│       └── ham10000/               # Dataset HAM10000 (não incluído)
├── reports/
│   ├── checkpoints/                # Modelos treinados (.pth)
│   └── figures/                    # Gráficos de avaliação
├── src/
│   ├── data/
│   │   ├── dataset_ham10000.py     # Dataset class do PyTorch
│   │   └── prepare_ham10000.py     # Script de preparação dos dados
│   ├── models/
│   │   └── model_resnet50.py       # Arquitetura do modelo
│   ├── train.py                    # Script de treinamento
│   ├── test.py                     # Script de teste e visualizações
│   └── utils/                      # Funções auxiliares
├── requirements.txt                # Dependências do projeto
└── README.md                       # Este arquivo
```

---

## 🚀 Instalação e Execução

### **Passo 1: Clonar o Repositório**

```bash
git clone https://github.com/seu-usuario/skin-cancer-cnn.git
cd skin-cancer-cnn
```

---

### **Passo 2: Criar e Ativar o Ambiente Virtual**

O uso de um **ambiente virtual** é essencial para isolar as dependências do projeto e evitar conflitos com outras instalações Python.

**Windows (PowerShell ou CMD):**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

> **Dica:** Você saberá que o ambiente está ativo quando aparecer `(.venv)` no início da linha do terminal.

---

### **Passo 3: Instalar as Dependências**

Com o ambiente virtual ativado, instale todas as bibliotecas necessárias:

```bash
pip install -r requirements.txt
```

Este comando instalará:
- PyTorch, torchvision
- Streamlit
- Albumentations, OpenCV
- Pandas, NumPy, Scikit-learn
- E outras dependências listadas

**Tempo estimado:** 5-10 minutos (depende da conexão de internet)

---

### **Passo 4: Baixar o Dataset HAM10000**

O dataset **HAM10000** não está incluído no repositório devido ao seu tamanho (~3GB). Você precisa baixá-lo manualmente:

1. **Acesse o Kaggle:**  
   👉 [HAM10000 Dataset - Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

2. **Faça o download dos arquivos:**
   - `HAM10000_metadata.csv`
   - `HAM10000_images_part_1.zip`
   - `HAM10000_images_part_2.zip`

3. **Descompacte e organize os arquivos:**

   Coloque os arquivos na seguinte estrutura:
   ```
   data/raw/ham10000/
   ├── HAM10000_metadata.csv
   ├── HAM10000_images_part_1/
   │   └── (5000 imagens .jpg)
   └── HAM10000_images_part_2/
       └── (5015 imagens .jpg)
   ```

> **Nota:** Certifique-se de que as pastas `HAM10000_images_part_1` e `HAM10000_images_part_2` contêm as imagens `.jpg` descompactadas.

---

### **Passo 5: Preparar os Dados**

Execute o script de preparação para processar o dataset e criar os splits de treino/validação/teste:

```bash
python src/data/prepare_ham10000.py
```

**O que este script faz:**
- ✅ Lê o arquivo `HAM10000_metadata.csv`
- ✅ Converte as 7 classes originais em classificação **binária** (Melanoma vs. Outras)
- ✅ Divide o dataset em **70% treino**, **15% validação**, **15% teste**
- ✅ Gera 3 arquivos CSV em `data/processed/`:
  - `train_metadata.csv` (7.010 imagens)
  - `val_metadata.csv` (1.502 imagens)
  - `test_metadata.csv` (1.503 imagens)

**Saída esperada:**
```
🚀 PREPARAÇÃO HAM10000 - Classificação Binária
✅ 10,015 imagens processadas
✅ Splits salvos em data/processed/
🎉 CONCLUÍDO!
```

---

### **Passo 6: (Opcional) Treinar o Modelo**

Se você deseja treinar seu próprio modelo do zero (ou retreinar):

1. **Configure os hiperparâmetros** em `configs/config.yaml`:
   ```yaml
   epochs: 30        # Número de épocas (recomendado: 20-30)
   batch_size: 32    # Tamanho do batch
   lr: 0.0003        # Learning rate
   ```

2. **Execute o treinamento:**
   ```bash
   python src/train.py
   ```

   **Tempo estimado:**
   - CPU: ~15-20 horas para 30 épocas
   - GPU (CUDA): ~2-3 horas para 30 épocas

3. **Acompanhe o progresso:**
   - O melhor modelo será salvo automaticamente em `reports/checkpoints/best_model.pth`
   - Métricas de treino e validação serão exibidas a cada época

> **Nota:** Um modelo pré-treinado já está incluído no repositório em `reports/checkpoints/best_model.pth` (AUC: 0.86, 2 épocas).

---

### **Passo 7: (Opcional) Avaliar o Modelo no Conjunto de Teste**

Para gerar métricas detalhadas e visualizações:

```bash
python src/test.py
```

**Este script gera:**
- 📊 Métricas completas (Accuracy, AUC-ROC, Sensitivity, Specificity, etc.)
- 📈 **ROC Curve** (`reports/figures/test_roc_curve.png`)
- 📈 **Precision-Recall Curve** (`reports/figures/test_pr_curve.png`)
- 🔢 **Confusion Matrix** (`reports/figures/test_confusion_matrix.png`)

---

### **Passo 8: Executar o Aplicativo Streamlit**

Agora você está pronto para usar a interface web! Execute:

```bash
streamlit run app.py
```

**O que acontece:**
- 🚀 O Streamlit abrirá automaticamente no navegador (geralmente `http://localhost:8501`)
- 📤 Faça upload de uma imagem dermatoscópica
- 🎚️ Ajuste o **threshold de decisão** (sensibilidade do modelo)
- 🔬 Veja a **probabilidade de melanoma** em tempo real
- ✅ Classificação: "Possível Melanoma" ou "Lesão Benigna"

> **Dica:** Para melhores resultados, use imagens **dermatoscópicas** (ampliadas, bem iluminadas, lesão centralizada). Fotos comuns de celular podem ter menor precisão.

---

## 📊 Resultados do Modelo

### **Métricas no Conjunto de Validação (2 épocas - teste inicial):**

| Métrica | Valor |
|---------|-------|
| **AUC-ROC** | 0.8640 |
| **Accuracy** | 84.55% |
| **Sensitivity (Recall)** | 59.28% |
| **Specificity** | 87.72% |
| **F1-Score** | 0.5456 |

### **Confusion Matrix (Validação - Época 2):**

|            | Predito: Benigna | Predito: Melanoma |
|------------|------------------|-------------------|
| **Real: Benigna** | 1171 (TN) | 165 (FP) |
| **Real: Melanoma** | 68 (FN) | 99 (TP) |

**Interpretação:**
- ✅ O modelo detectou **99 de 167 melanomas** (59.28% de sensibilidade)
- ✅ Classificou corretamente **1171 de 1336 lesões benignas** (87.72% de especificidade)
- 📈 Com mais épocas de treinamento (20-30), espera-se AUC > 0.90

---

## 🎯 Funcionalidades do Aplicativo

- 📤 **Upload de Imagens:** Suporta JPG, JPEG e PNG
- 🎚️ **Ajuste de Threshold:** Controle entre sensibilidade (detectar mais melanomas) e especificidade (evitar falsos alarmes)
- 🔬 **Análise em Tempo Real:** Probabilidade de melanoma exibida instantaneamente
- 📊 **Visualização Intuitiva:** Barra de progresso e alertas visuais
- ⚙️ **Informações do Modelo:** Sidebar com detalhes técnicos (arquitetura, dataset, métricas)

**Exemplo de uso:**

1. Faça upload de uma imagem de lesão de pele
2. Ajuste o threshold (padrão: 0.5):
   - **< 0.3:** Muito sensível (detecta mais, pode ter falsos alarmes)
   - **0.5:** Balanceado
   - **> 0.7:** Conservador (menos alarmes, pode perder melanomas)
3. Veja o resultado e a probabilidade calculada

---

## 🧪 Dataset: HAM10000

**Human Against Machine with 10000 training images**

- **Total:** 10.015 imagens dermatoscópicas
- **Classes originais (7):** mel, nv, bcc, akiec, bkl, df, vasc
- **Classificação binária:**
  - Melanoma (mel): 1.113 imagens (11.1%)
  - Outras lesões: 8.902 imagens (88.9%)
- **Fonte:** [Kaggle - HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **Referência:** Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Sci. Data* 5, 180161 (2018).

---

## 📈 Arquitetura do Modelo

**ResNet50 com Transfer Learning**

- **Backbone:** ResNet50 pré-treinado no ImageNet (1.4M imagens, 1000 classes)
- **Cabeça Customizada:**
  ```
  Linear(2048 → 256) → ReLU → Dropout(0.4) → Linear(256 → 1)
  ```
- **Loss Function:** BCEWithLogitsLoss com `pos_weight=8.0` (para balancear classes)
- **Otimizador:** Adam (lr=0.0003, weight_decay=0.01)
- **Augmentation:** Rotação, flips, brilho/contraste, shift/scale/rotate

**Tamanho do modelo:** ~95 milhões de parâmetros (~188 MB)

---

## 🔧 Configuração Avançada

### **Modificar Hiperparâmetros:**

Edite o arquivo `configs/config.yaml`:

```yaml
epochs: 30              # Número de épocas de treinamento
batch_size: 32          # Tamanho do batch (reduzir se ficar sem memória)
lr: 0.0003              # Learning rate
weight_decay: 0.01      # Regularização L2
img_size: 224           # Tamanho das imagens (224x224)
num_workers: 2          # Workers para DataLoader (0 no Windows se der erro)
```

### **Treinar em GPU (CUDA):**

Se você tem uma GPU NVIDIA com CUDA instalado, o script detectará automaticamente e usará a GPU, acelerando o treinamento em ~10x.

**Verificar disponibilidade:**
```python
import torch
print(torch.cuda.is_available())  # Deve retornar True
print(torch.cuda.get_device_name(0))  # Nome da sua GPU
```

---

## 📚 Referências e Links Úteis

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [ResNet Paper (He et al., 2015)](https://arxiv.org/abs/1512.03385)
- [HAM10000 Dataset Paper](https://www.nature.com/articles/sdata2018161)
- [Albumentations Documentation](https://albumentations.ai/docs/)
