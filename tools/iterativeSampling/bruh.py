import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from numpy.polynomial.polynomial import polyfit

# File paths
file_paths = [
    "f:/Programming/GitHub/chameleon-org/chameleon-testing/tools/iterativeSampling/5_BLEUandSentiment/gemini2FlashIterations_bleu.json",
    "f:/Programming/GitHub/chameleon-org/chameleon-testing/tools/iterativeSampling/5_BLEUandSentiment/gemini15ProIterations_bleu.json",
    "f:/Programming/GitHub/chameleon-org/chameleon-testing/tools/iterativeSampling/5_BLEUandSentiment/GPT4oIterations_bleu.json"
]

titles = ["Gemini 2.0 Flash", "Gemini 1.5 Pro", "GPT 4o"]

# Prepare data for plotting
def prepare_bleu_data(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)

    paraphrased_scores = []
    iterations_scores_dict = {}

    for entry in data:
        paraphrased_scores.append(entry["paraphrased_score"])

        for i, score in enumerate(entry["iterations_scores"], start=1):
            if i not in iterations_scores_dict:
                iterations_scores_dict[i] = []
            iterations_scores_dict[i].append(score)

    return paraphrased_scores, iterations_scores_dict

# Load data from all files
datasets = [prepare_bleu_data(file_path) for file_path in file_paths]

# Create subplots (3 columns for 3 models)
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for ax, (paraphrased_scores, iterations_scores_dict), title in zip(axes, datasets, titles):
    # Define x positions dynamically
    positions = [1] + list(range(2, 2 + len(iterations_scores_dict)))
    labels = ["Paraphrased"] + [f"Iter {i}" for i in sorted(iterations_scores_dict.keys())]
    data_to_plot = [paraphrased_scores] + [iterations_scores_dict[i] for i in sorted(iterations_scores_dict.keys())]

    # Plot the boxplot
    ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=False)
    
    # Add polynomial best fit and calculate R²
    x_values = []
    y_values = []
    for pos, data in zip(positions, data_to_plot):
        x_values.extend([pos] * len(data))
        y_values.extend(data)
    
    x_array = np.array(x_values)
    y_array = np.array(y_values)
    
    # Calculate polynomial regression (degree=2 for quadratic fit)
    polynomial_degree = 2
    coeffs = polyfit(x_array, y_array, polynomial_degree)
    
    # Generate polynomial function
    p = np.poly1d(np.flip(coeffs))
    
    # Calculate R² for polynomial fit
    y_predicted = p(x_array)
    r_squared = 1 - np.sum((y_array - y_predicted)**2) / np.sum((y_array - np.mean(y_array))**2)
    
    # Add polynomial regression curve
    x_line = np.linspace(min(positions), max(positions), 100)
    y_line = p(x_line)
    ax.plot(x_line, y_line, 'r--', linewidth=2)
    
    # Add R² value to the plot
    ax.text(0.95, 0.95, f'R² = {r_squared:.3f}', 
            transform=ax.transAxes, 
            verticalalignment='top', 
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Customize the plot
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(title)  # Each subplot gets its own title
    ax.set_ylabel("BLEU Score")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add a vertical separator after "Paraphrased"
    ax.axvline(x=1.5, color='black', linestyle='-', linewidth=0.8)

# Set the main title
plt.tight_layout()
plt.show()
