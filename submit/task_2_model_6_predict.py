import csv

import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration


 # Load the best model
model = T5ForConditionalGeneration.from_pretrained('/NLP/task2_ff')
tokenizer = AutoTokenizer.from_pretrained('/NLP/task2_ff', model_max_length = 512)
def process_test_dataset():
    jsonl_data = pd.read_json("test.jsonl", lines=True)
    jsonl_data['targetParagraphs'] = jsonl_data['targetParagraphs'].map(lambda x: '. '.join(x))
    jsonl_data['postText'] = jsonl_data['postText'].map(lambda x: x[0])
    jsonl_data['input'] = jsonl_data['postText'] + " Target paragraphs: " + jsonl_data['targetParagraphs']
    dataset = Dataset.from_pandas(jsonl_data)
    dataset = dataset.map(lambda example:{"input_text": example["input"]}
, batched=True)
    return dataset

test_dataset = process_test_dataset()


test_inp = tokenizer(test_dataset['input_text'], padding=True, truncation=True, max_length=512, return_tensors="pt")

model.eval()
with torch.no_grad():
    outputs = model.generate(input_ids=test_inp['input_ids'],min_length=1)

decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

# Save the predictions to a CSV file
with open("solution_t2_n.csv", "w", encoding='utf-8', newline='') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
    writer.writerow(["id", "spoiler"])
    for ind, pred in enumerate(decoded_outputs):
        if len(pred) <2:
            pred = "no spoiler"
        writer.writerow([ind, pred])
