from collections import Counter
import pandas as pd
import json 

# Reloading the ambiguous data points for analysis
file_path = 'ambiguous_data_points.jsonl'
data_points = []

with open(file_path, 'r') as file:
    for line in file:
        data_points.append(json.loads(line))

# Analyze label distribution
label_counts = Counter([point['label'] for point in data_points])

# Analyze recurring patterns in premise and hypothesis
premise_words = Counter()
hypothesis_words = Counter()

for point in data_points:
    premise_words.update(point['premise'].lower().split())
    hypothesis_words.update(point['hypothesis'].lower().split())

# Identify the most common words in premises and hypotheses
most_common_premise_words = premise_words.most_common(10)
most_common_hypothesis_words = hypothesis_words.most_common(10)

# Display the results
{
    "label_distribution": label_counts,
    "most_common_premise_words": most_common_premise_words,
    "most_common_hypothesis_words": most_common_hypothesis_words
}
print(label_counts)
print(most_common_hypothesis_words)
print(most_common_premise_words)