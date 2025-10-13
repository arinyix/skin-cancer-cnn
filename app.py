"App Streamlit - Detecção de Melanoma com Classificação Binária"

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==================== MODELO ResNet50 ====================
class SkinCancerResNet50(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        
        # Carrega ResNet50 pré-treinado
        self.base = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        
        # Número de features da última camada (2048)
        in_features = self.base.fc.in_features
        
        # Substitui a última camada para classificação binária
        self.base.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)  # Saída: 1 neurônio (binário)
        )

    def forward(self, x):
        # Passa a imagem pelo modelo.
        return self.base(x)
    
# ==================== FUNÇÕES AUXILIARES ====================
@st.cache_resource
def load_model():

    checkpoint_path = "reports/checkpoints/best_model.pth"

    model = SkinCancerResNet50(pretrained=False)

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    return model

def preprocess_image(image):
    image_np = np.array(image)

    if image_np.shape[-1] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    transformed = transform(image=image_np)
    image_tensor = transformed['image']

    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor

def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)

        probability = torch.sigmoid(output).item()

    return probability
    
# ==================== CONFIGURAÇÃO DA PÁGINA ====================
st.set_page_config(
    page_title="Detector de Melanoma",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Detector de Melanoma com CNN")
st.markdown(
"""
Este aplicativo usa **Deep Learning** para classificar lesões de pele.

**⚠️ AVISO IMPORTANTE:** Este é um projeto educacional. 
Não substitui avaliação médica profissional!
""")

st.divider()

# ==================== INTERFACE PRINCIPAL ====================
try:
    model = load_model()
    st.success("✅ Modelo carregado com sucesso!")
except Exception as e:
    st.error(f"❌ Erro ao carregar modelo: {e}")
    st.stop()

# SIDEBAR
st.sidebar.header("⚙️ Configurações")

# Slider de threshold
threshold = st.sidebar.slider(
    "🎚️ Threshold de Decisão",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Ajuste a sensibilidade do modelo. Valores menores aumentam a detecção de melanoma."
)

st.sidebar.markdown(f"""
**Threshold atual:** `{threshold:.2f}`

- **< 0.3**: Muito sensível (detecta mais, mais alarmes falsos)
- **0.5**: Balanceado
- **> 0.7**: Conservador (detecta menos, mais preciso)
""")

st.sidebar.divider()

# Informações do modelo
st.sidebar.markdown("""
### 📊 Sobre o Modelo

**Arquitetura:** ResNet50  
**Dataset:** HAM10000  
**Imagens de treino:** 7.010  
**Épocas:** 2 (teste)  
**AUC-ROC:** 0.86

---
**⚠️ Disclaimer:** Este é um projeto educacional.
""")

# ==================== ÁREA PRINCIPAL ====================
# Upload de imagem
st.subheader("📤 Upload de Imagem")
uploaded_file = st.file_uploader(
    "Escolha uma imagem de lesão de pele",
    type=["jpg", "jpeg", "png"],
    help="Formato aceitos: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Imagem Original")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🔬 Análise")

        with st.spinner("Analisando imagem.."):
            image_tensor = preprocess_image(image)
            probability = predict(model, image_tensor)
            is_melanoma = probability >= threshold

            st.metric(
                label="Probabilidade de Melanoma",
                value=f"{probability*100:.2f}%",
            )

            if is_melanoma:
                st.error("⚠️ Alerta: Possível MELANOMA detectado!")
                st.markdown("Procure um dematologista para avaliação.")

            else:
                st.success("✅ Lesão provavelmente BENIGNA.")
                st.markdown("Continue monitorando sua pele regularmente.")

            st.progress(probability)

            st.divider()
            st.caption(f"Threshold usado: {threshold:.2f} | Logit: {np.log(probability/(1-probability+1e-10)):.2f}")

else:
    st.info("👆 Faça o upload de uma imagem para começar a análise.")

# ==================== RODAPÉ ====================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <p style='margin: 8px 0; font-size: 15px;'>
            <strong>🎓 Universidade Federal do Oeste do Pará (UFOPA)</strong>
        </p>
        <p style='margin: 8px 0; font-size: 14px; color: #666;'>
            <strong>Discentes:</strong> Maria de Fátima Mota da Silva | Rayssa de Sousa Simões
        </p>
        <p style='margin: 15px 0 0 0; font-size: 12px; color: #888;'>
            ⚕️ Sistema para fins educacionais • Não substitui diagnóstico médico profissional
        </p>
    </div>
""", unsafe_allow_html=True)