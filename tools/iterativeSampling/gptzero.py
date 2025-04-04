import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Define the file path
file_path = "f:/Programming/GitHub/chameleon-org/chameleon-testing/tools/iterative/finalTextDataSets.json"

def process_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    # Create dictionaries to store data by model
    model_data = {
        "gemini-2.0-flash": {"original": {"ZeroGPT": [], "ROBERTA": [], "GPTZero": []}, 
                        "summarized": {"ZeroGPT": [], "ROBERTA": [], "GPTZero": []},
                        "iterations": {}},
        "gemini-1.5-pro": {"original": {"ZeroGPT": [], "ROBERTA": [], "GPTZero": []}, 
                       "summarized": {"ZeroGPT": [], "ROBERTA": [], "GPTZero": []},
                       "iterations": {}},
        "gpt-4o": {"original": {"ZeroGPT": [], "ROBERTA": [], "GPTZero": []}, 
                "summarized": {"ZeroGPT": [], "ROBERTA": [], "GPTZero": []},
                "iterations": {}}
    }
    
    def safe_append(score_list, value):
        if value is not None:
            score_list.append(value)
    
    # Process each entry in the dataset
    for entry in data:
        # Get original scores
        zerogpt_score = entry["originalText"]["detectionScores"].get("zerogpt")
        roberta_score = entry["originalText"]["detectionScores"].get("roberta")
        gptzero_score = entry["originalText"]["detectionScores"].get("gptzero")
        
        # For each model, get summarized scores and iteration scores
        for model_key in ["gemini-2.0-flash", "gemini-1.5-pro", "gpt-4o"]:
            if model_key not in entry["models"]:
                continue
                
            model_name_map = {
                "gemini-2.0-flash": "gemini-2.0-flash",
                "gemini-1.5-pro": "gemini-1.5-pro", 
                "gpt-4o": "gpt-4o"
            }
            
            mapped_model = model_name_map[model_key]
            
            # Add original scores to each model's data
            safe_append(model_data[mapped_model]["original"]["ZeroGPT"], zerogpt_score)
            safe_append(model_data[mapped_model]["original"]["ROBERTA"], roberta_score)
            safe_append(model_data[mapped_model]["original"]["GPTZero"], gptzero_score)
            
            # Get summarized text scores
            model_info = entry["models"][model_key]
            if "summarizedText" in model_info and "detectionScores" in model_info["summarizedText"]:
                safe_append(model_data[mapped_model]["summarized"]["ZeroGPT"], 
                            model_info["summarizedText"]["detectionScores"].get("zerogpt"))
                safe_append(model_data[mapped_model]["summarized"]["ROBERTA"], 
                            model_info["summarizedText"]["detectionScores"].get("roberta"))
                safe_append(model_data[mapped_model]["summarized"]["GPTZero"], 
                            model_info["summarizedText"]["detectionScores"].get("gptzero"))
            
            # Process iteration data
            if "iterations" in model_info:
                for i, iteration in enumerate(model_info["iterations"]):
                    # Skip empty iterations
                    if not iteration:
                        continue
                        
                    iteration_num = i + 1
                    if iteration_num not in model_data[mapped_model]["iterations"]:
                        model_data[mapped_model]["iterations"][iteration_num] = {"ZeroGPT": [], "ROBERTA": [], "GPTZero": []}
                    
                    if "detectionScores" in iteration:
                        safe_append(model_data[mapped_model]["iterations"][iteration_num]["ZeroGPT"], 
                                   iteration["detectionScores"].get("zerogpt"))
                        safe_append(model_data[mapped_model]["iterations"][iteration_num]["ROBERTA"], 
                                   iteration["detectionScores"].get("roberta"))
                        safe_append(model_data[mapped_model]["iterations"][iteration_num]["GPTZero"], 
                                   iteration["detectionScores"].get("gptzero"))
    
    return [
        (model_data["gemini-2.0-flash"]["original"], model_data["gemini-2.0-flash"]["summarized"], model_data["gemini-2.0-flash"]["iterations"]),
        (model_data["gemini-1.5-pro"]["original"], model_data["gemini-1.5-pro"]["summarized"], model_data["gemini-1.5-pro"]["iterations"]),
        (model_data["gpt-4o"]["original"], model_data["gpt-4o"]["summarized"], model_data["gpt-4o"]["iterations"])
    ]

# Process the data
datasets = process_data(file_path)

def plot_scores(fig, axes, method, datasets, titles):
    for i, (original_scores, summarized_scores, iterations_data) in enumerate(datasets):
        ax = axes[i]
        positions = [1, 2]
        labels = ["Original", "Summarized"]
        data_to_plot = [original_scores[method], summarized_scores[method]]
        pos_index = 3
        
        for iteration_num, scores in sorted(iterations_data.items()):
            positions.append(pos_index)
            labels.append(f"Iter {iteration_num}")
            data_to_plot.append(scores[method])
            pos_index += 1
        
        ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=False)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Detection Scores")
        ax.set_title(f"{titles[i]} - {method}")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.axvline(x=1.5, color='black', linestyle='-', linewidth=0.8)
        
        # Perform linear regression excluding the first 'Original' data point
        # Flatten data and positions for regression
        scores_flat = []
        positions_flat = []
        
        # Start from Summarized (index 1)
        for idx, scores_list in enumerate(data_to_plot[1:], start=1):
            scores_flat.extend(scores_list)
            positions_flat.extend([positions[idx]] * len(scores_list))
        
        if len(set(positions_flat)) > 1:
            # Calculate linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(positions_flat, scores_flat)
            r_squared = r_value**2
            
            # Plot the line of best fit
            x_line = np.array([min(positions_flat), max(positions_flat)])
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, 'r--', alpha=0.7)
            
            # Display R² value in top right corner
            ax.text(0.95, 0.95, f"R² = {r_squared:.3f}", transform=ax.transAxes, 
                    verticalalignment='top', horizontalalignment='right', fontsize=12, 
                    bbox=dict(facecolor='white', alpha=0.8))

# Create figure for ZeroGPT
titles = ["Gemini 2.0 Flash", "Gemini 1.5 Pro", "GPT 4o"]
fig1, axes1 = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), sharex=True)
plot_scores(fig1, axes1, "ZeroGPT", datasets, titles)
plt.tight_layout()
plt.show()

# Create figure for ROBERTA
fig2, axes2 = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), sharex=True)
plot_scores(fig2, axes2, "ROBERTA", datasets, titles)
plt.tight_layout()
plt.show()

# Create figure for GPTZero
fig3, axes3 = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), sharex=True)
plot_scores(fig3, axes3, "GPTZero", datasets, titles)
plt.tight_layout()
plt.show()
