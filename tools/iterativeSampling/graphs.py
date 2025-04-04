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

def calculate_best_fit(x_data, y_data):
    """Calculate the line of best fit for the data points."""
    # Skip if we don't have enough data
    if len(x_data) < 2 or len(y_data) < 2:
        return None, None, None
    
    # Calculate the line of best fit using linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
    
    # Generate points for the line
    x_fit = np.array([min(x_data), max(x_data)])
    y_fit = slope * x_fit + intercept
    
    return x_fit, y_fit, r_value**2  # Return r-squared value

def plot_scores(fig, axes, datasets, titles):
    for i, (original_scores, summarized_scores, iterations_data) in enumerate(datasets):
        ax = axes[i]
        positions = [1, 2]
        labels = ["Original", "Summarized"]
        data_to_plot = [original_scores, summarized_scores]
        pos_index = 3
        
        for iteration_num, scores in sorted(iterations_data.items()):
            positions.append(pos_index)
            labels.append(f"Iter {iteration_num}")
            data_to_plot.append(scores)
            pos_index += 1
        
        ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=False)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Coherence Scores")
        ax.set_title(titles[i])
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.axvline(x=1.5, color='black', linestyle='-', linewidth=0.8)
        
        # Add line of best fit
        x_data = []
        y_data = []
        # Start from summarized (position 2) for clearer trend analysis
        for idx, (position, scores_group) in enumerate(zip(positions[1:], data_to_plot[1:])):
            x_data.extend([position] * len(scores_group))
            y_data.extend(scores_group)
        
        if x_data and y_data:
            x_fit, y_fit, r_squared = calculate_best_fit(x_data, y_data)
            if x_fit is not None:
                ax.plot(x_fit, y_fit, 'r--', linewidth=2, label=f'Best fit (R²={r_squared:.3f})')
                ax.legend(loc='upper right')

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
        pos_index = 2
        
        for iteration_num, scores in sorted(iterations_data.items()):
            positions.append(pos_index)
            labels.append(f"Iter {iteration_num}")
            data_to_plot.append(scores)
            pos_index += 1
        
        ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=False)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Coherence Scores")
        ax.set_title(titles[i])
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add line of best fit
        x_data = []
        y_data = []
        for idx, (position, scores_group) in enumerate(zip(positions, data_to_plot)):
            x_data.extend([position] * len(scores_group))
            y_data.extend(scores_group)
        
        if x_data and y_data:
            x_fit, y_fit, r_squared = calculate_best_fit(x_data, y_data)
            if x_fit is not None:
                ax.plot(x_fit, y_fit, 'r--', linewidth=2, label=f'Best fit (R²={r_squared:.3f})')
                ax.legend(loc='upper right')

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
