import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.models.model_resnet50 import SkinCancerResNet50
import cv2, numpy as np

def visualize_gradcam(img_path="data/processed/test/example.jpg"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SkinCancerResNet50(pretrained=False)
    model.load_state_dict(torch.load("reports/checkpoints/best_model.pth", map_location=device))
    model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    image = Image.open(img_path).convert("RGB")
    rgb_img = np.array(image) / 255.0
    tensor = preprocess(image).unsqueeze(0).to(device)

    cam = GradCAMPlusPlus(model=model, target_layers=[model.base.layer4[-1]])
    grayscale_cam = cam(input_tensor=tensor)[0, :]
    cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    plt.imshow(cam_img)
    plt.title("Grad-CAM — Foco do Modelo")
    plt.axis("off")
    plt.savefig("reports/figures/gradcam_example.png")
    print("✅ Grad-CAM salva em reports/figures/gradcam_example.png")

if __name__ == "__main__":
    visualize_gradcam()
