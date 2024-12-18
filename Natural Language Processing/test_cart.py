import datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines  # To create custom legend entries
import seaborn as sns
import json

def main():
    # Load SNLI training dataset
    dataset = datasets.load_dataset("snli")["train"]

    # Remove unlabeled examples and subset 25,000 samples
    dataset = dataset.filter(lambda ex: ex['label'] != -1)  # Remove unlabeled examples
    if len(dataset) > 25000:
        dataset = dataset.shuffle(seed=42).select(range(25000))  # Randomly shuffle and select 25,000 samples

    # Add an index to the dataset to uniquely identify each example
    dataset = dataset.map(lambda example, idx: {"idx": idx}, with_indices=True)

    # Load pretrained model checkpoint and tokenizer
    model_checkpoint = "trained_model/checkpoint-206013"
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define optimizer and DataLoader
    optimizer = AdamW(model.parameters(), lr=5e-5)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Store metrics for each example
    confidence_records = {i: [] for i in range(len(dataset))}  # To track confidence for each example

    model.train()
    for epoch in range(3):
        print(f"Starting epoch {epoch+1}")
        for batch in train_loader:
            # Move inputs and labels to device
            inputs = tokenizer(batch['premise'], batch['hypothesis'], return_tensors='pt', padding=True, truncation=True).to(device)
            labels = batch['label'].to(device)
            idxs = batch['idx']

            # Forward pass
            outputs = model(**inputs, labels=labels)

            # Loss and backward propagation
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Get confidence scores (probabilities of correct labels)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            correct_probs = probs[range(len(labels)), labels]

            # Update confidence records for each example in the batch
            for idx, prob in zip(idxs, correct_probs):
                confidence_records[idx.item()].append(prob.item())

    print("Finished training. Calculating data map...")
    data_map = []
    for idx, confidences in confidence_records.items():
        if len(confidences) > 0:
            avg_confidence = np.mean(confidences)
            variability = np.std(confidences)
            data_map.append({'index': idx, 'avg_confidence': avg_confidence, 'variability': variability})

    # Check if data map is empty
    if not data_map:
        print("Data map is empty. No plot will be generated.")
        return

    # Convert data_map to numpy arrays for easier handling
    mean_probs = np.array([point['avg_confidence'] for point in data_map])
    std_devs = np.array([point['variability'] for point in data_map])

    # Smoothened categorization for easy-to-learn, ambiguous, and hard-to-learn
    # Defining overlapping regions to create a gradient-like categorization

    easy_to_learn = [
        point for point in data_map
        if point['avg_confidence'] > 0.75 and point['variability'] < 0.2
    ]

    hard_to_learn = [
        point for point in data_map
        if point['avg_confidence'] < 0.45 and point['variability'] < 0.3
    ]

    # For ambiguous points, overlap with both easy-to-learn and hard-to-learn thresholds
    ambiguous = [
        point for point in data_map
        if (0.45 <= point['avg_confidence'] <= 0.75) or (0.2 <= point['variability'] <= 0.5)
    ]

    # Debug: Print the number of points in each category to confirm distribution
    print(f"Number of easy-to-learn points: {len(easy_to_learn)}")
    print(f"Number of hard-to-learn points: {len(hard_to_learn)}")
    print(f"Number of ambiguous points: {len(ambiguous)}")

     # Find Index and return sentences into .jsonl file
    # Save ambiguous points to a .jsonl file
    ambiguous_points = []
    for i in range(len(ambiguous)):
        if ambiguous[i]:
            ambiguous_points.append(dataset[i])  # Append the original data point

    output_path = "ambiguous_data_points.jsonl"
    with open(output_path, "w") as f:
        for point in ambiguous_points:
            f.write(json.dumps(point) + "\n")

    print(f"Ambiguous data points saved to '{output_path}'.")

     # Find Index and return sentences into .jsonl file
    # Save ambiguous points to a .jsonl file
    hard_points = []
    for i in range(len(hard_to_learn)):
        if hard_to_learn[i]:
            hard_points.append(dataset[i])  # Append the original data point

    output_path = "hard_data_points.jsonl"
    with open(output_path, "w") as f:
        for point in hard_points:
            f.write(json.dumps(point) + "\n")

    print(f"Hard data points saved to '{output_path}'.")

     # Find Index and return sentences into .jsonl file
    # Save ambiguous points to a .jsonl file
    easy_points = []
    for i in range(len(easy_to_learn)):
        if easy_to_learn[i]:
            easy_points.append(dataset[i])  # Append the original data point

    output_path = "easy_data_points.jsonl"
    with open(output_path, "w") as f:
        for point in ambiguous_points:
            f.write(json.dumps(point) + "\n")

    print(f"Easy data points saved to '{output_path}'.")

    # Create boolean masks for categorization
    easy_mask = np.array([point in easy_to_learn for point in data_map])
    hard_mask = np.array([point in hard_to_learn for point in data_map])
    ambiguous_mask = np.array([point in ambiguous for point in data_map])

    # Plotting the scatter plot
    plt.figure(figsize=(10, 8))

    # Define correctness levels and custom bins
    custom_correctness_levels = [0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0]  # Explicit correctness levels
    correctness_labels = np.digitize(mean_probs, custom_correctness_levels) - 1  # Map mean_probs to custom bins

    # Define marker styles for correctness levels
    markers = ['o', '*', 's', 'd', '^', 'v', 'P']  # Circle, star, square, diamond, triangle up, triangle down, plus
    colors = {'hard_to_learn': 'blue', 'easy_to_learn': 'red', 'ambiguous': 'green'}

    # Create legend entries for custom correctness levels
    legend_entries = []
    for level, marker in zip(custom_correctness_levels, markers):
        legend_entries.append(
            mlines.Line2D(
                [], [], color='black', marker=marker, linestyle='None', markersize=8, label=f"{level:.1f}"
            )
        )

    # Plot each group (hard-to-learn, easy-to-learn, ambiguous) with custom correctness levels
    for group, mask, label_color in [
        ('hard_to_learn', hard_mask, colors['hard_to_learn']),
        ('easy_to_learn', easy_mask, colors['easy_to_learn']),
        ('ambiguous', ambiguous_mask, colors['ambiguous']),
    ]:
        for i, marker in enumerate(markers):
            group_mask = (correctness_labels == i) & mask
            plt.scatter(
                std_devs[group_mask],
                mean_probs[group_mask],
                c=label_color,
                marker=marker,
                alpha=0.6,
                s=25,  # Set smaller marker size
                edgecolors='white',  # Add white border around points
                linewidth=0.5  # Set width of the border
            )

    # Add the custom legend for correctness levels
    plt.legend(
        handles=legend_entries,
        title="correct.",
        loc="upper right",
        fontsize=10,
        title_fontsize=11,
        frameon=True,
        shadow=False,
        edgecolor="black"
    )

    # Highlight KDE for ambiguous points (optional)
    sns.kdeplot(x=std_devs[ambiguous_mask], y=mean_probs[ambiguous_mask], cmap="Greens", fill=True, alpha=0.3)
    sns.kdeplot(x=std_devs[hard_mask], y=mean_probs[hard_mask], cmap="Blues", fill=True, alpha=0.3)
    sns.kdeplot(x=std_devs[easy_mask], y=mean_probs[easy_mask], cmap="Reds", fill=True, alpha=0.3)

    # Add labels to regions
    bb = lambda c: dict(boxstyle="round,pad=0.3", ec=c, lw=2, fc="white")
    plt.text(0.02, 0.85, "easy-to-learn", fontsize=12, bbox=bb('red'))
    plt.text(0.02, 0.18, "hard-to-learn", fontsize=12, bbox=bb('blue'))
    plt.text(0.31, 0.5, "ambiguous", fontsize=12, bbox=bb('green'))

    # Add labels, grid, and title
    plt.xlabel("Variability (Standard Deviation)", fontsize=12)
    plt.ylabel("Confidence (Mean Probability)", fontsize=12)
    plt.title("SNLI-ELECTRA-small Data Map", fontsize=14)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
