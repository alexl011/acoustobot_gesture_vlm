import os
import json
import torch
import open_clip
import time
from datetime import datetime
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, optim

# --- 1. Configuration ---
# Note: These are now default values. They will be saved in a config.json for each run.
CONFIG = {
    "annotations_file": './annotations.txt',
    "image_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset')),
    "runs_dir": os.path.join(os.path.dirname(__file__), 'training_runs'),
    "model_name": 'ViT-B-32-quickgelu',
    "pretrained_dataset": 'openai',
    "num_epochs": 50,
    "batch_size": 5,
    "learning_rate": 1e-3,
    "validation_split": 0.2,
    "random_seed": 42
}

# --- 2. Custom Dataset for Feature Extraction ---
class GestureFeatureDataset(Dataset):
    def __init__(self, annotations_file, img_dir, preprocess_func):
        self.img_paths = []
        self.labels = []
        self.class_to_idx = {}
        idx_counter = 0

        # Check if annotation file exists
        if not os.path.exists(annotations_file):
            raise FileNotFoundError(f"Annotations file not found at {annotations_file}. Please run create_annotations.py first.")

        with open(annotations_file, 'r') as f:
            for line in f:
                img_name, label = line.strip().split('\t')
                full_path = os.path.join(img_dir, img_name)
                if os.path.exists(full_path):
                    self.img_paths.append(full_path)
                else:
                    print(f"Warning: Image not found at {full_path}. Skipping.")
                    continue
                
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

# --- 3. Feature Extraction Function ---
def extract_features(model, data_loader, device):
    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            features = model.encode_image(images)
            all_features.append(features)
            all_labels.append(labels)
    
    return torch.cat(all_features), torch.cat(all_labels)

# --- 4. Training Logic ---
def main():
    # --- Part 0: Setup Run Directory ---
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(CONFIG['runs_dir'], run_timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"--- Starting new training run: {run_timestamp} ---")
    print(f"Archiving results in: {run_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(CONFIG['random_seed'])
    print(f"Using device: {device}")

    # --- Part 1: Load CLIP model and prepare data ---
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        CONFIG['model_name'], pretrained=CONFIG['pretrained_dataset'], device=device
    )
    clip_model.eval()

    # Create the full dataset
    full_dataset = GestureFeatureDataset(CONFIG['annotations_file'], CONFIG['image_dir'], preprocess)
    
    # Split dataset into training and validation sets
    num_data = len(full_dataset)
    num_val = int(CONFIG['validation_split'] * num_data)
    num_train = num_data - num_val
    train_dataset, val_dataset = random_split(full_dataset, [num_train, num_val])

    print(f"Dataset split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    # Create DataLoaders for both sets
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # --- Part 2: Extract features for both sets ---
    print("Extracting features with CLIP for training and validation sets...")
    train_features, train_labels = extract_features(clip_model, train_loader, device)
    val_features, val_labels = extract_features(clip_model, val_loader, device)
    print(f"Feature extraction complete.")
    print(f"Training features shape: {train_features.shape}")
    print(f"Validation features shape: {val_features.shape}")

    # --- Part 3: Training the Linear Probe with Validation ---
    probe_model = nn.Linear(train_features.shape[1], len(full_dataset.class_to_idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(probe_model.parameters(), lr=CONFIG['learning_rate'])

    print("\n--- Starting Model Training ---")
    training_metrics = [] # To store metrics for each epoch

    for epoch in range(CONFIG['num_epochs']):
        epoch_start_time = time.time()
        
        # --- Training Phase ---
        probe_model.train()
        optimizer.zero_grad()
        
        train_outputs = probe_model(train_features)
        train_loss = criterion(train_outputs, train_labels)
        
        train_loss.backward()
        optimizer.step()
        
        # --- Validation Phase ---
        probe_model.eval()
        with torch.no_grad():
            val_outputs = probe_model(val_features)
            val_loss = criterion(val_outputs, val_labels)
            
            # Calculate validation accuracy
            _, predicted = torch.max(val_outputs, 1)
            correct_predictions = (predicted == val_labels).sum().item()
            total_samples = val_labels.size(0)
            val_accuracy = (correct_predictions / total_samples) * 100

        epoch_duration = time.time() - epoch_start_time
        
        # --- Logging ---
        print(
            f"Epoch [{epoch+1}/{CONFIG['num_epochs']}] | "
            f"Train Loss: {train_loss.item():.4f} | "
            f"Val Loss: {val_loss.item():.4f} | "
            f"Val Acc: {val_accuracy:.2f}% | "
            f"Time: {epoch_duration:.2f}s"
        )
        
        # Store metrics
        training_metrics.append({
            'epoch': epoch + 1,
            'train_loss': train_loss.item(),
            'val_loss': val_loss.item(),
            'val_accuracy': val_accuracy,
            'epoch_duration_sec': epoch_duration
        })

    # --- Part 4: Save Metrics and Final Model ---
    # Define path for the metrics archive
    metrics_output_path = os.path.join(run_dir, 'training_metrics.json')

    print(f"\nTraining complete.")
    
    with open(metrics_output_path, 'w') as f:
        json.dump(training_metrics, f, indent=4)
    print(f"Run metrics saved to '{metrics_output_path}'")
    
    # --- Part 5: Update the main model files for direct use ---
    script_dir = os.path.dirname(__file__)
    main_probe_path = os.path.join(script_dir, 'gesture_model.pt')
    main_mapping_path = os.path.join(script_dir, 'class_mapping.json')
    
    torch.save(probe_model.state_dict(), main_probe_path)
    print(f"Latest model updated at: '{main_probe_path}'")
    
    # Re-save the class mapping to the main directory as well
    with open(main_mapping_path, 'w') as f:
        json.dump(full_dataset.idx_to_class, f)

    # --- Part 6: Save Run Details ---
    # Count images per class from the annotations file for this run's record
    gesture_counts = {}
    with open(CONFIG['annotations_file'], 'r') as f:
        for line in f:
            if line.strip():
                _, label = line.strip().split('\t')
                gesture_counts[label] = gesture_counts.get(label, 0) + 1
    
    run_details = {
        'run_timestamp': run_timestamp,
        'config': CONFIG,
        'dataset_statistics': {
            'total_images': len(full_dataset),
            'training_images': len(train_dataset),
            'validation_images': len(val_dataset),
            'gesture_counts': gesture_counts
        }
    }
    details_output_path = os.path.join(run_dir, 'run_details.json')
    with open(details_output_path, 'w') as f:
        json.dump(run_details, f, indent=4)
    print(f"Run details saved to '{details_output_path}'")

if __name__ == "__main__":
    main() 