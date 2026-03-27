import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Classes
all_classes = ['No Finding', 'Infiltration', 'Effusion', 'Atelectasis',
               'Nodule', 'Mass', 'Pneumothorax', 'Consolidation',
               'Pleural_Thickening', 'Cardiomegaly', 'Emphysema',
               'Edema', 'Fibrosis', 'Pneumonia', 'Hernia']

# Load model
device = torch.device('cpu')
model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, 15)
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
model = model.to(device)
model.eval()

# Transform
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

target_layer = model.features[-1]

def add_caption(img_np, disease, confidence):
    img_pil = Image.fromarray(img_np)
    new_img = Image.new('RGB', (img_pil.width, img_pil.height + 30), (255, 255, 255))
    new_img.paste(img_pil, (0, 0))
    draw = ImageDraw.Draw(new_img)
    draw.text((6, img_pil.height + 6), f"{disease} — {confidence:.1%}", fill=(30, 30, 30))
    return new_img

def predict(image):
    if image is None:
        return {}, []

    img_rgb = image.convert('RGB')
    img_resized = img_rgb.resize((224, 224))
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    input_tensor = inference_transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output).cpu().numpy()[0]

    results = {all_classes[i]: float(probs[i]) for i in range(15)}

    detected = [(all_classes[i], probs[i]) for i in range(15) if probs[i] >= 0.5]
    if len(detected) == 0:
        top2_idx = np.argsort(probs)[-2:][::-1]
        detected = [(all_classes[i], probs[i]) for i in top2_idx]

    gradcam_images = []
    cam = GradCAM(model=model, target_layers=[target_layer])

    for disease_name, confidence in detected:
        class_idx = all_classes.index(disease_name)
        targets = [ClassifierOutputTarget(class_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        captioned = add_caption(visualization, disease_name, confidence)
        gradcam_images.append(captioned)

    return results, gradcam_images

with gr.Blocks(title="Chest X-Ray Classifier") as demo:
    gr.Markdown("# <h1 align='center'>Chest X-Ray Disease Classifier</h1>")
    gr.Markdown("<h4 align='center'>DenseNet-121 · NIH CXR8 · 112,120 images · Mean AUC 0.8309 · 14 pathologies</h4>")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type='pil', label="Upload Chest X-Ray")
            submit_btn = gr.Button("Analyze", variant="primary")
            gr.Markdown("⚠️ *Research use only. Not for clinical diagnosis.*")
        with gr.Column():
            label_output = gr.Label(num_top_classes=6, label="Disease Probabilities")

    gr.Markdown("### Grad-CAM Heatmaps")
    gr.Markdown("*One heatmap per detected disease. Red = high attention, blue = low attention.*")

    gradcam_gallery = gr.Gallery(label="", columns=3, rows=2, height=400)

    submit_btn.click(fn=predict, inputs=image_input, outputs=[label_output, gradcam_gallery])

demo.launch()
