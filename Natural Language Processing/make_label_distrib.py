import json
from collections import Counter
import matplotlib.pyplot as plt

# Define a function to count labels in a file
def count_labels(file_path):
    label_counter = Counter()
    with open(file_path, 'r') as file:
        for line in file:
            # Parse the dictionary
            example = json.loads(line.strip())
            # Increment the label counter
            label_counter[example['label']] += 1
    return label_counter
def count_labels2(file_path):
    label_counter = Counter()
    with open(file_path, 'r') as file:
        for line in file:
            # Parse the dictionary
            example = json.loads(line.strip())
            # Increment the label counter
            label_counter[example['label']] += 1
    return label_counter
# File paths
file2 = 'snli_validation_examples_modified.jsonl'
file1 = 'snli_validation_examples.txt'

# Count labels for each file
counts_file1 = count_labels(file1)
counts_file2 = count_labels2(file2)

# Display results
print("Label counts in file 1:")
for label in range(0, 3):  # Labels 0, 1, 2
    print(f"Label {label}: {counts_file1.get(label, 0)}")

print("\nLabel counts in file 2:")
for label in range(0, 3):  # Labels 0, 1, 2
    print(f"Label {label}: {counts_file2.get(label, 0)}")

# Prepare data for visualization
labels = ['Label 0', 'Label 1', 'Label 2']
counts1 = [counts_file1.get(i, 0) for i in range(3)]
counts2 = [counts_file2.get(i, 0) for i in range(3)]

# Create bar chart
x = range(len(labels))  # Position of bars

plt.figure(figsize=(8, 6))  # ACL standard size
plt.bar(x, counts1, width=0.4, label='Original', color='steelblue', align='center')
plt.bar([pos + 0.4 for pos in x], counts2, width=0.4, label='Contrast', color='#c9d6df', align='center')

# Customize chart for ACL standards
plt.xticks([pos + 0.2 for pos in x], labels, fontsize=12)  # Add labels at center of grouped bars
plt.xlabel('Labels', fontsize=14)
plt.ylabel('Counts', fontsize=14)
plt.title('Label Counts', fontsize=16)
plt.legend(fontsize=12)

# Adjust layout
plt.tight_layout()

# Save plot to a file in high resolution (suitable for ACL submission)
plt.savefig('label_counts_acl.png', dpi=300)

# Show plot
plt.show()
