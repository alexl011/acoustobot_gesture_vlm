import cv2
import torch
import open_clip
import json
import os
from PIL import Image
from torch import nn

# --- 1. Configuration ---
script_dir = os.path.dirname(__file__)
probe_model_path = os.path.join(script_dir, 'gesture_model.pt')
mapping_path = os.path.join(script_dir, 'class_mapping.json')
clip_model_name = 'ViT-B-32-quickgelu'
pretrained_dataset = 'openai'
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. Load Models and Mapping ---
def load_models():
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
    
    return clip_model, probe_model, preprocess, class_mapping

# --- 3. Real-time Classification Loop ---
def main():
    clip_model, probe_model, preprocess, class_mapping = load_models()
    print("Models loaded. Starting webcam...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        # Convert the frame to a PIL Image for preprocessing
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Preprocess the image for CLIP
        input_img = preprocess(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            # Step 1: Extract features using CLIP
            image_features = clip_model.encode_image(input_img)
            
            # Step 2: Classify features using the linear probe
            logits = probe_model(image_features)
            
            # Get the prediction
            probabilities = logits.softmax(dim=-1)
            top_prob, top_idx = probabilities.topk(1, dim=-1)

            predicted_class_idx = str(top_idx.item())
            predicted_label = class_mapping.get(predicted_class_idx, "Unknown")
            confidence = top_prob.item()

        # Draw the prediction on the frame
        label_text = f"Gesture: {predicted_label} ({confidence:.2f})"
        cv2.putText(frame, label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Real-Time Gesture Classification', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 