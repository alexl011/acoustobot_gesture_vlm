import json
import matplotlib.pyplot as plt
import os
import sys
import argparse

# --- Main plotting logic ---
def main():
    """
    Loads training metrics from a specific run directory and generates plots.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Generate plots from a training metrics file."
    )
    parser.add_argument(
        "run_id", 
        type=str, 
        help="The timestamped ID of the training run to plot (e.g., '2024-07-30_10-30-00')."
    )
    args = parser.parse_args()
    
    run_id = args.run_id
    script_dir = os.path.dirname(__file__)
    run_dir = os.path.join(script_dir, 'training_runs', run_id)

    metrics_file = os.path.join(run_dir, 'training_metrics.json')
    plots_dir = os.path.join(run_dir, 'plots') # Save plots inside the specific run's folder
    loss_plot_file = os.path.join(plots_dir, 'loss_curve.png')
    accuracy_plot_file = os.path.join(plots_dir, 'accuracy_curve.png')

    print(f"Attempting to generate plots for run: {run_id}")

    # Check if the run directory and metrics file exist
    if not os.path.isdir(run_dir):
        print(f"Error: Run directory not found at '{run_dir}'")
        sys.exit(1)
        
    if not os.path.exists(metrics_file):
        print(f"Error: Metrics file not found at '{metrics_file}'.")
        print("Ensure the training for this run was completed successfully.")
        sys.exit(1)

    # Load the metrics data
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    # Create the directory for saving plots if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Plots will be saved to: {plots_dir}")

    # Extract data for plotting
    epochs = [m['epoch'] for m in metrics]
    train_losses = [m['train_loss'] for m in metrics]
    val_losses = [m['val_loss'] for m in metrics]
    val_accuracies = [m['val_accuracy'] for m in metrics]

    # --- Load Dataset Info from run_details.json ---
    details_file = os.path.join(run_dir, 'run_details.json')
    dataset_info = "Dataset images: unavailable"
    if os.path.exists(details_file):
        with open(details_file, 'r') as f:
            details = json.load(f)
            gesture_counts = details.get('dataset_statistics', {}).get('gesture_counts', {})
            if gesture_counts:
                dataset_info = f"Dataset images: " + ", ".join([f"{k}: {v}" for k, v in gesture_counts.items()])
    else:
        print(f"Warning: run_details.json not found for run {run_id}. Dataset info will be unavailable.")

    # --- Plot 1: Training and Validation Loss ---
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, color='b', linestyle='-', label='Training Loss')
    plt.plot(epochs, val_losses, color='r', linestyle='-', label='Validation Loss')
    plt.title(f'Training and Validation Loss\n(Run: {run_id})\n{dataset_info}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_plot_file)
    print(f"Loss curve plot saved to: {loss_plot_file}")
    plt.close()

    # --- Plot 2: Validation Accuracy ---
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_accuracies, color='g', linestyle='-', label='Validation Accuracy')
    plt.title(f'Validation Accuracy\n(Run: {run_id})\n{dataset_info}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 105) # Set y-axis from 0 to 105 for consistency
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(accuracy_plot_file)
    print(f"Accuracy curve plot saved to: {accuracy_plot_file}")
    plt.close()
    
    print("\nPlot generation complete.")

if __name__ == '__main__':
    main() 