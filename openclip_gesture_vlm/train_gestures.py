import os
import json
import torch
import open_clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

# --- 1. Configuration ---
annotations_file = './annotations.txt'
image_dir = '../dataset'
output_probe_path = './gesture_model.pt'
output_mapping_path = './class_mapping.json'

model_name = 'ViT-B-32-quickgelu'
pretrained_dataset = 'openai'
num_epochs = 50
batch_size = 5
learning_rate = 1e-3 # A higher learning rate is fine for this small model

# --- 2. Custom Dataset for Feature Extraction ---
class GestureFeatureDataset(Dataset):
    def __init__(self, annotations_file, img_dir, preprocess_func):
        self.img_paths = []
        self.labels = []
        self.class_to_idx = {}
        idx_counter = 0

        with open(annotations_file, 'r') as f:
            for line in f:
                img_name, label = line.strip().split('\t')
                self.img_paths.append(os.path.join(img_dir, img_name))
                
                if label not in self.class_to_idx:
                    self.class_to_idx[label] = idx_counter
                    idx_counter += 1
                self.labels.append(self.class_to_idx[label])
        
        self.preprocess = preprocess_func
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        processed_image = self.preprocess(image)
        return processed_image, torch.tensor(label, dtype=torch.long)

# --- 3. Training Logic ---
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Part 1: Feature Extraction ---
    # Load the pretrained CLIP model
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained_dataset, device=device
    )
    clip_model.eval() # Ensure CLIP is in evaluation mode and frozen

    # Create the dataset and dataloader
    dataset = GestureFeatureDataset(annotations_file, image_dir, preprocess)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Save the class mapping for the classifier
    with open(output_mapping_path, 'w') as f:
        json.dump(dataset.idx_to_class, f)
    print(f"Class mapping saved to {output_mapping_path}")

    # Extract features from all images
    print("Extracting features with CLIP...")
    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            features = clip_model.encode_image(images)
            all_features.append(features)
            all_labels.append(labels)
            
    # Concatenate all features and labels
    train_features = torch.cat(all_features).to(device)
    train_labels = torch.cat(all_labels).to(device)
    print(f"Feature extraction complete. Tensor shape: {train_features.shape}")

    # --- Part 2: Training the Linear Probe ---
    # Define the simple linear classifier
    probe_model = nn.Linear(train_features.shape[1], len(dataset.class_to_idx)).to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(probe_model.parameters(), lr=learning_rate)

    print("Training the linear probe...")
    probe_model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = probe_model(train_features)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Save the trained linear probe
    torch.save(probe_model.state_dict(), output_probe_path)
    print(f"Training complete. Linear probe model saved to '{output_probe_path}'")

if __name__ == "__main__":
    main() 