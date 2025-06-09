import os
import json
import csv
import glob

def analyze_model_performance(base_path):
    """
    Analyzes performance metrics for Qwen models based on output files,
    correctly using VRAM for memory and averaging latency.

    Args:
        base_path (str): The path to the directory containing the model output folders.
    """
    output_path = os.path.join(base_path, 'qwen2.5-output')
    if not os.path.exists(output_path):
        print(f"Error: Directory not found at {output_path}")
        return {}
        
    model_dirs = sorted(os.listdir(output_path))
    results = {}

    for model_dir_name in model_dirs:
        if not model_dir_name.startswith('Qwen_Qwen2.5-VL'):
            continue

        # --- Extract Model Size for dictionary key ---
        try:
            model_size = model_dir_name.split('-')[2]
            model_key = f"Qwen2.5-VL\n{model_size}"
        except IndexError:
            print(f"Warning: Could not parse model size from directory: {model_dir_name}")
            continue

        results[model_key] = {
            "Memory Usage (GB)": "N/A",
            "Latency (ms/input)": "N/A",
            "Accuracy (0.xx)": "--" # Accuracy cannot be determined from the files
        }
        
        current_model_path = os.path.join(output_path, model_dir_name)

        # --- 1. Calculate Peak VRAM Usage from resource_usage.csv ---
        # This correctly uses GPU VRAM, not system memory.
        # For the 32B model, this assumes the CSV logs the combined VRAM of both GPUs.
        try:
            resource_file = os.path.join(current_model_path, 'resource_usage.csv')
            peak_vram_mb = 0
            with open(resource_file, 'r') as f:
                # Use DictReader to handle potential whitespace in headers
                reader = csv.DictReader(f)
                headers = [h.strip() for h in reader.fieldnames]
                reader.fieldnames = headers

                for row in reader:
                    vram_mb = float(row['memory_used_mb'])
                    if vram_mb > peak_vram_mb:
                        peak_vram_mb = vram_mb
            
            # Convert Peak VRAM from MB to GB
            peak_vram_gb = peak_vram_mb / 1024
            results[model_key]["Memory Usage (GB)"] = f"{peak_vram_gb:.2f}"

        except (FileNotFoundError, ValueError, KeyError, TypeError) as e:
            print(f"Warning: Could not process VRAM from resource_usage.csv for {model_key}. Error: {e}")

        # --- 2. Calculate Average Latency from JSON files ---
        # This averages the inference time per video file (per input).
        try:
            json_files = glob.glob(os.path.join(current_model_path, 'video_*_analysis.json'))
            total_duration_s = 0
            video_count = len(json_files)

            if video_count > 0:
                for json_file in json_files:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        # Get the total inference time for one video
                        total_duration_s += data.get('performance_metrics', {}).get('inference_duration_seconds', 0)
                
                # Calculate average latency per video (input) and convert seconds to milliseconds
                avg_latency_ms = (total_duration_s / video_count) * 1000
                results[model_key]["Latency (ms/input)"] = f"{avg_latency_ms:.2f}"

        except (json.JSONDecodeError, ZeroDivisionError) as e:
            print(f"Warning: Could not process latency from JSON files for {model_key}. Error: {e}")

    return results

def print_latex_table(results):
    """Prints the results in a filled-out LaTeX table format."""
    
    model_order = ["Qwen2.5-VL\n3B", "Qwen2.5-VL\n7B", "Qwen2.5-VL\n32B"]
    
    header = r"""
\begin{table}[h]
\centering
\resizebox{0.7\textwidth}{!}{%
\begin{tabular}{||c c c c c||} 
 \hline
 & \makecell{VIBA-Net\\(Ours)} & \makecell{Qwen2.5-VL\\3B} & \makecell{Qwen2.5-VL\\7B} & \makecell{Qwen2.5-VL\\32B} \\ 
 \hline\hline"""
    
    footer = r"""\end{tabular}
}
\caption{Comparison of latency, memory usage, and parameter size across models.}
\end{table}"""
    
    viba_net_data = {"Memory Usage (GB)": "0.04", "Latency (ms/input)": "--", "Accuracy (0.xx)": "--", "Parameters (M)": "3.6"}
    qwen_params = {"3B": "3000", "7B": "7000", "32B": "32000"}

    print("--- Generated LaTeX Table Code ---")
    print(header)
    
    # Memory Usage Row
    mem_row = f" Memory Usage (GB) & {viba_net_data['Memory Usage (GB)']}"
    for model in model_order:
        mem_row += f" & {results.get(model, {}).get('Memory Usage (GB)', '--')}"
    print(mem_row + " \\\\ \n \\hline")

    # Latency Row
    lat_row = f" Latency (ms/input) & {viba_net_data['Latency (ms/input)']}"
    for model in model_order:
        lat_row += f" & {results.get(model, {}).get('Latency (ms/input)', '--')}"
    print(lat_row + " \\\\ \n \\hline")

    # Accuracy Row
    acc_row = f" Accuracy (0.xx) & {viba_net_data['Accuracy (0.xx)']}"
    for model in model_order:
        acc_row += f" & {results.get(model, {}).get('Accuracy (0.xx)', '--')}"
    print(acc_row + " \\\\ \n \\hline")
    
    # Parameters Row
    param_row = f" Parameters (M) & {viba_net_data['Parameters (M)']}"
    for model in model_order:
        size_key = model.split('\n')[1]
        param_row += f" & {qwen_params.get(size_key, '--')}"
    print(param_row + " \\\\ \n \\hline")

    print(footer)
    print("\nNOTE: Accuracy is marked as '--' as it cannot be calculated from the provided data.")
    print("NOTE: Memory Usage is Peak VRAM reported in resource_usage.csv.")
    print("NOTE: Latency is the average inference duration per video file.")


if __name__ == '__main__':
    # The script uses an absolute path, so it can be run from anywhere.
    # It points to /home/seunguk/Desktop/CSE493_project/models
    project_base_path = os.path.expanduser('~/Desktop/CSE493_project/models')
    
    if not os.path.exists(project_base_path):
        print(f"Error: The specified path does not exist: {project_base_path}")
        print("Please update the 'project_base_path' variable in the script if needed.")
    else:
        analysis_results = analyze_model_performance(project_base_path)
        print_latex_table(analysis_results)
