from datasets import load_dataset
import json
import random

# Load the SNLI dataset
dataset = load_dataset('snli')

# Select the validation set
validation_data = dataset['validation']

# Filter out examples with label -1
filtered_data = [example for example in validation_data if example['label'] != -1]

# Number of examples to extract
num_examples = 600

# Ensure reproducibility by setting a random seed
random.seed(42)

# Randomly sample 600 examples from the filtered data
sampled_data = random.sample(filtered_data, min(num_examples, len(filtered_data)))

# Open a text file for writing
with open('snli_validation_examples.txt', 'w', encoding='utf-8') as f:
    # Extract and write each example in the specified format
    for example in sampled_data:
        example_dict = {
            'premise': example['premise'],
            'hypothesis': example['hypothesis'],
            'label': example['label']  # Labels: 0 = entailment, 1 = neutral, 2 = contradiction
        }
        # Convert the dictionary to a JSON string and write it to the file
        f.write(json.dumps(example_dict, ensure_ascii=False) + '\n')

print(f"Saved {len(sampled_data)} examples to 'snli_validation_examples.txt'")
