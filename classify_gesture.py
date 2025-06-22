import sys
import json
import torch
import open_clip
from PIL import Image
from torch import nn

# --- 1. Configuration ---
probe_model_path = './gesture_model.pt'
mapping_path = './class_mapping.json'
clip_model_name = 'ViT-B-32-quickgelu'
pretrained_dataset = 'openai'

# --- 2. Inference Logic ---
def classify_gesture(image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load Models and Mapping ---
    # Load the class mapping
    with open(mapping_path, 'r') as f:
        class_mapping = json.load(f)
    num_classes = len(class_mapping)

    # Load the frozen CLIP model for feature extraction
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=pretrained_dataset, device=device
    )
    clip_model.eval()

    # Load the trained linear probe
    probe_model = nn.Linear(512, num_classes) # 512 is the feature dim for ViT-B/32
    probe_model.load_state_dict(torch.load(probe_model_path, map_location=device))
    probe_model.to(device)
    probe_model.eval()

    # --- Prepare Image ---
    try:
        image = Image.open(image_path).convert("RGB")
        preprocessed_image = preprocess(image).unsqueeze(0).to(device)
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'")
        return

    # --- Run Inference ---
    with torch.no_grad():
        # Step 1: Extract features using CLIP
        image_features = clip_model.encode_image(preprocessed_image)
        
        # Step 2: Classify features using the linear probe
        logits = probe_model(image_features)
        
        # Get the prediction
        probabilities = logits.softmax(dim=-1)
        top_prob, top_idx = probabilities.topk(1, dim=-1)

        predicted_class_idx = str(top_idx.item())
        predicted_label = class_mapping.get(predicted_class_idx, "Unknown")
        confidence = top_prob.item()

    print(f"Predicted gesture: {predicted_label} (Confidence: {confidence:.2f})")
    return predicted_label

# --- 3. Main Execution Block ---
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python classify_gesture.py <path_to_image>")
    else:
        image_file = sys.argv[1]
        classify_gesture(image_file) 