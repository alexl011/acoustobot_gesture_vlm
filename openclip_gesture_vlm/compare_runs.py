import json
import matplotlib.pyplot as plt
import os
import sys

def create_line_plot(all_run_metrics, metric_key, title, ylabel, save_path):
    """Generic function to create a line plot comparing a metric across all runs."""
    plt.figure(figsize=(12, 7))
    
    for run_data in all_run_metrics:
        epochs = [m['epoch'] for m in run_data['metrics']]
        values = [m[metric_key] for m in run_data['metrics']]
        plt.plot(epochs, values, 'o-', label=run_data['run_label'])
        
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")
    plt.close()

def create_accuracy_bar_chart(run_results, save_path):
    """Creates a bar chart specifically for comparing best accuracy."""
    sorted_data = sorted(run_results, key=lambda x: x['best_accuracy'], reverse=True)
    run_labels = [r['run_label'].replace('_', '\n') for r in sorted_data]
    values = [r['best_accuracy'] for r in sorted_data]
    
    plt.figure(figsize=(max(6, len(run_labels) * 1.5), 7))
    bars = plt.bar(run_labels, values, color='skyblue')
    
    plt.ylabel('Best Validation Accuracy (%)')
    plt.title('Comparison of Best Validation Accuracy Across Runs')
    plt.ylim(0, 105)
    plt.xticks(rotation=0)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')
        
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")
    plt.close()

def main():
    """
    Scans all training runs, compares their metrics, and generates
    a summary table and comparative plots.
    """
    script_dir = os.path.dirname(__file__)
    runs_dir = os.path.join(script_dir, 'training_runs')
    
    if not os.path.isdir(runs_dir):
        print(f"Error: Training runs directory not found at '{runs_dir}'")
        sys.exit(1)
        
    all_runs = sorted([d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))])
    
    if not all_runs:
        print(f"No training runs found in '{runs_dir}'.")
        sys.exit(0)
        
    print(f"Found {len(all_runs)} training runs. Comparing results...")
    
    all_run_metrics = []
    for run_id in all_runs:
        metrics_file = os.path.join(runs_dir, run_id, 'training_metrics.json')
        details_file = os.path.join(runs_dir, run_id, 'run_details.json')
        
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Load details to get dataset counts and create a run label
            gesture_counts = {}
            total_images = 0
            if os.path.exists(details_file):
                with open(details_file, 'r') as f:
                    details = json.load(f)
                    stats = details.get('dataset_statistics', {})
                    gesture_counts = stats.get('gesture_counts', {})
                    total_images = stats.get('total_images', 0)

            # Create a more descriptive label
            run_label = f"Number of dataset images = {total_images}"

            if metrics:
                all_run_metrics.append({
                    'run_id': run_id,
                    'run_label': run_label,
                    'metrics': metrics,
                    'best_accuracy': max(m['val_accuracy'] for m in metrics),
                    'gesture_counts': gesture_counts
                })
            else:
                print(f"Warning: Metrics for run '{run_id}' are empty. Skipping.")
        else:
            print(f"Warning: Could not find metrics for run '{run_id}'. Skipping.")
            
    if not all_run_metrics:
        print("No valid metrics found to compare.")
        sys.exit(0)
        
    # --- Print Summary Table (focused on best accuracy) ---
    sorted_for_table = sorted(all_run_metrics, key=lambda x: x['best_accuracy'], reverse=True)
    print("\n--- Training Run Comparison ---")
    header = f"{'Rank':<5} | {'Run':<18} | {'Best Validation Accuracy':<25} | {'Dataset Counts'}"
    print(header)
    print("-" * len(header))
    for i, result in enumerate(sorted_for_table):
        acc_str = f"{result['best_accuracy']:.2f}%"
        counts_str = ", ".join([f"{k}: {v}" for k, v in result['gesture_counts'].items()]) if result['gesture_counts'] else "N/A"
        row = f"{i+1:<5} | {result['run_label']:<18} | {acc_str:<25} | {counts_str}"
        print(row)
    print("-" * len(header))

    # --- Generate and Save Plots ---
    print("\nGenerating comparison plots...")
    # Bar chart for best accuracy
    create_accuracy_bar_chart(all_run_metrics, os.path.join(script_dir, 'comparison_accuracy.png'))
    # Line plot for training loss
    create_line_plot(all_run_metrics, 'train_loss', 'Comparison of Training Loss', 'Training Loss', os.path.join(script_dir, 'comparison_training_loss.png'))
    # Line plot for validation loss
    create_line_plot(all_run_metrics, 'val_loss', 'Comparison of Validation Loss', 'Validation Loss', os.path.join(script_dir, 'comparison_validation_loss.png'))
    # Line plot for epoch duration (speed)
    create_line_plot(all_run_metrics, 'epoch_duration_sec', 'Comparison of Training Speed', 'Epoch Duration (seconds)', os.path.join(script_dir, 'comparison_training_speed.png'))

    print("\nComparison complete.")

if __name__ == '__main__':
    main() 