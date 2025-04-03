import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Define the file paths
file_paths = [
    r"f:\Programming\GitHub\chameleon-org\chameleon-testing\tools\iterativeSampling\3_checkingHumanity\data\gemini2FlashIterationsWithScores.json",
    r"f:\Programming\GitHub\chameleon-org\chameleon-testing\tools\iterativeSampling\3_checkingHumanity\data\gemini15ProIterationsWithScores.json",
    r"f:\Programming\GitHub\chameleon-org\chameleon-testing\tools\iterativeSampling\3_checkingHumanity\data\GPT4oIterationsWithScores.json"
]

def process_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    iterations_data = {}
    original_scores = []
    summarized_scores = []
    
    def safe_append(score_list, value):
        if value is not None:
            score_list.append(value)
    
    for entry in data:
        safe_append(original_scores, entry.get("original_coherence_score"))
        safe_append(summarized_scores, entry.get("paraphrased_score"))
        
        for iteration in entry.get("iterations", []):
            iteration_num = iteration["iteration_number"]
            if iteration_num not in iterations_data:
                iterations_data[iteration_num] = []
            safe_append(iterations_data[iteration_num], iteration.get("coherence_score"))
    
    return original_scores, summarized_scores, iterations_data

def plot_scores(fig, axes, datasets, titles):
    for i, (original_scores, summarized_scores, iterations_data) in enumerate(datasets):
        ax = axes[i]
        positions = [1, 2]
        labels = ["Original", "Summarized"]
        data_to_plot = [original_scores, summarized_scores]
        iteration_labels = []  # Labels for Kendall's tau test (excluding 'Original')
        pos_index = 3
        
        for iteration_num, scores in sorted(iterations_data.items()):
            positions.append(pos_index)
            labels.append(f"Iter {iteration_num}")
            data_to_plot.append(scores)
            iteration_labels.extend([pos_index] * len(scores))  # Track positions for Kendall's tau
            pos_index += 1
        
        ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=False)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Coherence Scores")
        ax.set_title(titles[i])
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.axvline(x=1.5, color='black', linestyle='-', linewidth=0.8)
        
        # Perform Kendall's Tau test excluding the first 'Original' x-tick
        scores_flat = [score for group in data_to_plot[1:] for score in group]  # Exclude Original
        iter_labels_flat = [2] * len(data_to_plot[1]) + iteration_labels  # Start from Summarized (2)
        
        if len(set(iter_labels_flat)) > 1:
            tau, p_value = stats.kendalltau(iter_labels_flat, scores_flat)
            ax.text(0.95, 0.95, f"p = {p_value:.3f}", transform=ax.transAxes, 
                    verticalalignment='top', horizontalalignment='right', fontsize=12, 
                    bbox=dict(facecolor='white', alpha=0.8))

# Function to process coherence data files
def process_coherence_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    iterations_data = {}
    summarized_scores = []
    
    for entry in data:
        # Add summarized (paraphrased) coherence score if it exists and isn't None
        if "paraphrased_score" in entry and entry["paraphrased_score"] is not None:
            summarized_scores.append(entry["paraphrased_score"])
        
        # Process iteration coherence scores
        if "iterations_scores" in entry:
            iteration_scores = entry["iterations_scores"]
            for i, score in enumerate(iteration_scores):
                if score is not None:  # Skip None values
                    iteration_num = i + 1  # Assuming 1-indexed iterations
                    if iteration_num not in iterations_data:
                        iterations_data[iteration_num] = []
                    iterations_data[iteration_num].append(score)
    
    return summarized_scores, iterations_data

def plot_coherence_graphs():
    # Define the coherence file paths - updated to correct location
    coherence_file_paths = [
        r"f:\Programming\GitHub\chameleon-org\chameleon-testing\tools\iterativeSampling\6_Coherence\gemini2Flash_coherence.json",
        r"f:\Programming\GitHub\chameleon-org\chameleon-testing\tools\iterativeSampling\6_Coherence\gemini15Pro_coherence.json",
        r"f:\Programming\GitHub\chameleon-org\chameleon-testing\tools\iterativeSampling\6_Coherence\GPT4o_coherence.json"
    ]

    # Process coherence data
    coherence_data = [process_coherence_file(file_path) for file_path in coherence_file_paths]

    # Create figure for Coherence
    titles = ["Gemini 2.0 Flash", "Gemini 1.5 Pro", "GPT 4o"]
    fig3, axes3 = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), sharex=True)

    for i, (summarized_scores, iterations_data) in enumerate(coherence_data):
        ax = axes3[i]
        positions = [1]  # Start with position 1 for summarized
        labels = ["Summarized"]
        data_to_plot = [summarized_scores]
        iteration_labels = []
        pos_index = 2
        
        for iteration_num, scores in sorted(iterations_data.items()):
            positions.append(pos_index)
            labels.append(f"Iter {iteration_num}")
            data_to_plot.append(scores)
            iteration_labels.extend([pos_index] * len(scores))
            pos_index += 1
        
        ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=False)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Coherence Scores")
        ax.set_title(titles[i])
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Perform Kendall's Tau test if there's enough data
        if len(data_to_plot) > 1:  # Make sure we have more than just summarized data
            scores_flat = [score for group in data_to_plot for score in group]
            iter_labels_flat = [1] * len(data_to_plot[0])  # Start with summarized (1)
            for idx, group in enumerate(data_to_plot[1:], start=2):  # Add iteration labels
                iter_labels_flat.extend([idx] * len(group))
            
            if len(set(iter_labels_flat)) > 1 and len(scores_flat) > 1:
                tau, p_value = stats.kendalltau(iter_labels_flat, scores_flat)
                ax.text(0.95, 0.95, f"p = {p_value:.3f}", transform=ax.transAxes, 
                        verticalalignment='top', horizontalalignment='right', fontsize=12, 
                        bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

def main():
    datasets = [process_file(file_path) for file_path in file_paths]
    
    titles = ["Gemini 2.0 Flash", "Gemini 1.5 Pro", "GPT 4o"]
    
    # Create figure for Coherence Scores
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), sharex=True)
    plot_scores(fig, axes, datasets, titles)
    plt.tight_layout()
    plt.show()
    
    # Plot coherence graphs
    plot_coherence_graphs()

if __name__ == "__main__":
    main()
