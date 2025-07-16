import torch
import open_clip
import json
from torch import nn

# --- 1. Configuration ---
probe_model_path = './gesture_model.pt'
mapping_path = './class_mapping.json'
clip_model_name = 'ViT-B-32-quickgelu'
pretrained_dataset = 'openai'
onnx_export_path = 'gesture_model.onnx'

def main():
    device = 'cpu'
    print('Loading class mapping...')
    with open(mapping_path, 'r') as f:
        class_mapping = json.load(f)
    num_classes = len(class_mapping)
    print(f'Number of classes: {num_classes}')

    print('Loading CLIP model...')
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=pretrained_dataset, device=device
    )
    clip_model.eval()

    print('Loading trained linear probe...')
    probe_model = nn.Linear(512, num_classes)
    probe_model.load_state_dict(torch.load(probe_model_path, map_location=device))
    probe_model.eval()

    # Wrap both models in a single nn.Module
    class GestureClassifier(nn.Module):
        def __init__(self, clip_model, probe_model):
            super().__init__()
            self.clip_model = clip_model
            self.probe_model = probe_model
        def forward(self, image):
            features = self.clip_model.encode_image(image)
            logits = self.probe_model(features)
            return logits

    print('Wrapping models...')
    combined_model = GestureClassifier(clip_model, probe_model)
    combined_model.eval()

    # Create dummy input (batch size 1, 3x224x224)
    dummy_input = torch.randn(1, 3, 224, 224)

    print(f'Exporting to ONNX: {onnx_export_path}')
    torch.onnx.export(
        combined_model,
        dummy_input,
        onnx_export_path,
        input_names=['input'],
        output_names=['logits'],
        opset_version=17,  # Changed from 11 to 13
        dynamic_axes={'input': {0: 'batch_size'}, 'logits': {0: 'batch_size'}}
    )
    print('Export complete! ONNX model saved as gesture_model.onnx')

if __name__ == '__main__':
    main() 