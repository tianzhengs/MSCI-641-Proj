import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

import pandas as pd
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data
def load_data(file_path):
    data = []
    with open(r"C:\Tianzheng\OneDrive - University of Waterloo\24 S\MSCI 641\project\task-1\\"+file_path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    return df

# Prepare data
def prepare_data(df, test=False):
    df['postText'] = df['postText'].apply(lambda x: x[0])
    df['targetParagraphsVec'] = df['targetParagraphs'].apply(lambda x: ' '.join(x))
    if not test:
        df['tags'] = df['tags'].apply(lambda x: {'multi': 0, 'passage': 1, 'phrase': 2}[x[0]])
    return df


train_df = load_data(r"train.jsonl")
val_df = load_data("val.jsonl")
test_df = load_data("test.jsonl")


train_df = prepare_data(train_df)
val_df = prepare_data(val_df)
test_df = prepare_data(test_df, test=True)

# Split features and labels
x_train, y_train = train_df['postText'], train_df['tags']
x_val, y_val = val_df['postText'], val_df['tags']
x_test = test_df['postText']

# BERT model
# bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
# bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

bert_preprocess = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
bert_encoder = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"




def build_classifier_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    # preprocessing_layer = bert_preprocess(text_input)
    # encoder_inputs = preprocessing_layer['input_word_ids'], preprocessing_layer['input_mask'], preprocessing_layer['input_type_ids']
    # outputs = bert_encoder(encoder_inputs)

    preprocessing_layer = hub.KerasLayer(bert_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(bert_encoder, trainable=True, name='BERT_encoder')

    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(128, activation='relu')(net)
    net = tf.keras.layers.Dense(3, activation='softmax', name='classifier')(net)
    return tf.keras.Model(text_input, net)

classifier_model = build_classifier_model()

# Compile the model
loss = tf.keras.losses.SparseCategoricalCrossentropy()
metrics = tf.metrics.SparseCategoricalAccuracy()

epochs = 4
steps_per_epoch = tf.data.experimental.cardinality(tf.data.Dataset.from_tensor_slices((x_train, y_train))).numpy()
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

classifier_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)

# Train the model
history = classifier_model.fit(x_train, y_train,
                               validation_data=(x_val, y_val),
                               epochs=epochs)

# Evaluate the model
loss, accuracy = classifier_model.evaluate(x_val, y_val)
print(f'Validation loss: {loss}')
print(f'Validation accuracy: {accuracy}')

# Make predictions
y_pred = classifier_model.predict(x_test)
y_pred = tf.argmax(y_pred, axis=1).numpy()

# Save predictions
backdict = {0: 'multi', 1: 'passage', 2: 'phrase'}
with open("solution.csv", "w") as f:
    f.write("id,spoilerType\n")
    for ind, pre in enumerate(y_pred):
        f.write(f"{ind},{backdict[pre]}\n")

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 8))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['sparse_categorical_accuracy'], label='accuracy')
    plt.plot(history.history['val_sparse_categorical_accuracy'], label='val_accuracy')
    plt.ylim([0, 1])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')

    plt.show()

plot_history(history)

# Validation loss: 0.9807175993919373
# Validation accuracy: 0.5074999928474426

# with open('history.pickle', 'wb') as handle:
#     pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
#