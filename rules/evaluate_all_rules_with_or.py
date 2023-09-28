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

or_rules = []
with open('or_rules.txt', 'r') as file:
    # Read all lines and remove any trailing newline characters
    or_rules = [line.rstrip('\n') for line in file]

# array that contains tuples (rule, nr_matches, rule_eval_array)
all_rules_eval = []
for rule in tqdm(or_rules):
    # contains a tuple (y_true, y_pred)
    rule_eval = []
    nr_matches = 0
    for text, label, _ in test:
        match = re.search(rule, text, re.IGNORECASE)
        pred = 0
        if match:
            pred = 1
            nr_matches = nr_matches + 1
        rule_eval.append((label, pred))
    all_rules_eval.append((rule, nr_matches, rule_eval))


columns = ['rules', 'nr_matches', 'accuracy', 'f1', 'precision', 'recall', 'tn', 'fp', 'fn', 'tp']
or_rules_eval_df = pd.DataFrame(columns=columns)

for rule, nr_matches, rule_eval in tqdm(all_rules_eval):

    gt = []
    rule_pred = []

    [gt.append(y_true) for y_true, _ in rule_eval]
    [rule_pred.append(y_pred) for _, y_pred in rule_eval]

    accuracy = accuracy_score(gt, rule_pred)
    f1 = f1_score(gt, rule_pred)
    precision = precision_score(gt, rule_pred)
    recall = recall_score(gt, rule_pred)
    # rounded_accuracy = round(accuracy, 2)
    # rounded_f1 = round(f1, 2)
    # rounded_precision = round(precision, 2)
    # rounded_recall = round(recall, 2)
    tn, fp, fn, tp = confusion_matrix(gt, rule_pred).ravel()

    row = pd.Series([rule, nr_matches, round(accuracy, 2), round(f1, 2), round(precision, 2), round(recall, 2), tn, fp, fn, tp], index=columns)
    or_rules_eval_df = or_rules_eval_df.append(row, ignore_index=True)

or_rules_eval_df_sorted = or_rules_eval_df.sort_values(by='nr_matches', ascending=False)
or_rules_eval_df_sorted.to_csv('or_rules_eval_df_sorted.csv')