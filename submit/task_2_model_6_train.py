import torch
import numpy as np
import pandas as pd
import transformers
from transformers import TrainingArguments, Trainer
from datasets import Dataset, load_metric
from transformers import AutoTokenizer, T5ForConditionalGeneration
# import matplotlib.pyplot as plt

tokenizer = AutoTokenizer.from_pretrained("t5-small", model_max_length = 512)
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def process_dataset(name):
    jsonl_data = pd.read_json(name, lines=True)
    jsonl_data['tags'] = jsonl_data['tags'].apply(lambda x: x[0])
    jsonl_data['output'] = jsonl_data['spoiler'].map(lambda x: ' /n/ '.join(x))
    jsonl_data['targetParagraphs'] = jsonl_data['targetParagraphs'].map(lambda x: '. '.join(x))
    jsonl_data['postText'] = jsonl_data['postText'].map(lambda x: x[0])
    jsonl_data['input'] = jsonl_data['targetParagraphs'] + " " + jsonl_data['postText']
    dataset = Dataset.from_pandas(jsonl_data)
    dataset = dataset.map(lambda example:{"input_ids": tokenizer(example["input"], padding="max_length", truncation=True, max_length= 500, return_tensors="pt").input_ids, "labels": tokenizer(example["output"], padding="max_length", truncation=True, max_length= 100, return_tensors="pt").input_ids}
, batched=True)
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ['input_ids', 'labels']])
    return dataset


train_dataset = process_dataset('train.jsonl')
val_dataset = process_dataset('val.jsonl')

bleu_metric = load_metric("bleu", trust_remote_code=True)

def metrics(eval_pred):
    ll, labels = eval_pred
    predicted = torch.argmax(torch.tensor(ll[0]), dim=-1).tolist()
    predicted = tokenizer.batch_decode(predicted, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    predictions = [p.split() for p in predicted]
    labels = [[l.split()] for l in labels]
    score = bleu_metric.compute(predictions=predictions, references=labels)["bleu"]
    return { "bleu": score}


args = TrainingArguments(
    output_dir="/NLP/task2_results",
    num_train_epochs=5,
    per_device_train_batch_size=2,  # Decreased from 8 to 2
    per_device_eval_batch_size=2, # Decreased from 8 to 2
    gradient_accumulation_steps=8,  # Increased from 2 to 4
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=200,
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    learning_rate=3e-5,
    weight_decay=0.02,
    warmup_steps=1000,
    save_strategy="steps",
    save_total_limit=3,
    fp16=True,
    gradient_checkpointing=True
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=metrics
)

# List to save evaluation metrics
eval_metrics = []

# Callback to log metrics
class LogMetricsCallback(transformers.TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            eval_metrics.append((state.global_step, metrics["eval_bleu"]))

# Add the callback to the trainer
trainer.add_callback(LogMetricsCallback())



trainer.train()


print(trainer.evaluate())

# # Plot the metrics
# steps, bleus = zip(*eval_metrics)
# plt.plot(steps, bleus, marker='o')
# plt.xlabel("Steps")
# plt.ylabel("BLEU Score")
# plt.title("BLEU Score during Training")
# plt.grid()
# plt.show()