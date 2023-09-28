import pandas as pd
import numpy as np
import re
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm

dataset = pd.read_csv('./dataset/dataset.csv')

data = list(zip(dataset['id'], dataset['text'], dataset['subtask_a'], []))
with open('../dataset/post_id_divisions.json', 'r') as json_file:
    post_id_divisions = json.load(json_file)

train_ids = [int(x) for x in post_id_divisions['train']]
val_ids = [int(x) for x in post_id_divisions['val']]
test_ids = [int(x) for x in post_id_divisions['test']]

train = [(text, label, rationales) for id, text, label, rationales in data if id in train_ids]
val = [(text, label, rationales) for id, text, label, rationales in data if id in val_ids]
test = [(text, label, rationales) for id, text, label, rationales in data if id in test_ids]

and_rules = []
with open('../and_rules.txt', 'r') as file:
    # Read all lines and remove any trailing newline characters
    and_rules = [line.rstrip('\n') for line in file]

print('Evaluate on test set...')
# evaluate rules on val set
test_eval = []
for text, label, _ in tqdm(test):
    match = False
    new_label = 0
    for rule in and_rules:
        match = re.search(rule, text, re.IGNORECASE)
        if match:
            new_label = 1
            break
    test_eval.append((text, label, new_label))

y_true_test = [label for text, label, new_label in test_eval]
y_pred_test = [new_label for text, label, new_label in test_eval]

# Calculate precision
precision_test = precision_score(y_true_test, y_pred_test)

# Calculate recall
recall_test = recall_score(y_true_test, y_pred_test)

# Calculate F1 score
f1_test = f1_score(y_true_test, y_pred_test)
accuracy_test = accuracy_score(y_true_test, y_pred_test)

print('Test set results for and rules:')
print("Precision:", precision_test)
print("Recall:", recall_test)
print("Accuracy:", accuracy_test)
print("F1 Score:", f1_test)