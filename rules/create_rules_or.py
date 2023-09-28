import pandas as pd
import numpy as np
import re
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm



def convert_to_bool_array(x):
    if pd.notna(x):
        return np.array(x.split(";")).astype(bool)
    else:
        return []

def calculate_majority(row):
    if len(row) > 0:
        stacked_arrays = np.vstack((row['rationales_1'], row['rationales_2'], row['rationales_3']))
        summed_array = np.sum(stacked_arrays, axis=0)
        majority_voted_array = summed_array >= 2
        return majority_voted_array
    else:
        return []

def generate_and_rule_from_sentence(sentence: str, rationales) -> str:
    tokens = sentence.split()
    selected_tokens = [token for token, rationale in zip(tokens, rationales) if rationale]
    if len(selected_tokens) == 0:
        return ''
    rule = r"(?=.*?\b" + r"\b)(?=.*?\b".join(map(re.escape, selected_tokens)) + r"\b)"
    return rule

dataset = pd.read_csv('./dataset/dataset.csv')
rationales = pd.read_csv('./dataset/dataset_rationales.csv')
data_final = pd.merge(dataset, rationales, on="id", how="left")

data_final["rationales_1"] = data_final["rationales_1"].map(lambda x: convert_to_bool_array(x))
data_final["rationales_2"] = data_final["rationales_2"].map(lambda x: convert_to_bool_array(x))
data_final["rationales_3"] = data_final["rationales_3"].map(lambda x: convert_to_bool_array(x))

data_final['rationales'] = data_final.apply(calculate_majority, axis=1)

data = list(zip(data_final['id'], data_final['text'], data_final['subtask_a'], data_final['rationales']))

with open('./dataset/post_id_divisions.json', 'r') as json_file:
    post_id_divisions = json.load(json_file)

train_ids = [int(x) for x in post_id_divisions['train']]
val_ids = [int(x) for x in post_id_divisions['val']]
test_ids = [int(x) for x in post_id_divisions['test']]

train = [(text, label, rationales) for id, text, label, rationales in data if id in train_ids]
val = [(text, label, rationales) for id, text, label, rationales in data if id in val_ids]
test = [(text, label, rationales) for id, text, label, rationales in data if id in test_ids]

# use only train data to create rules
train_hate = [(text, label, rationales) for text, label, rationales in train if label == 1]

and_rules = []
for text, label, rationales in train_hate:
    rule = generate_and_rule_from_sentence(text, rationales)
    if rule == '':
        continue
    and_rules.append(rule)

with open('and_rules.txt', 'w') as file:
    for rule in and_rules:
        file.write(rule + '\n')