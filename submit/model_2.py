import numpy as np
import pandas as pd
import json

import gensim.downloader as api


import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch import save

wv = api.load('word2vec-google-news-300')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
def save_model(model, filepath):
    save(model.state_dict(), filepath)

def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set the model to evaluation mode
    return model

class TextClassifier(nn.Module):
    def __init__(self, activation_fn):
        super(TextClassifier, self).__init__()

        if activation_fn == 'ReLU':
            self.activation = nn.ReLU()
        elif activation_fn == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_fn == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('Invalid activation function')

        input_dim = 600
        hidden_dim = 300
        output_dim = 3

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def train_model(model, train_loader, val_loader, num_epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # L2 regularization
    minimum_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Validation Loss: {val_loss / len(val_loader)}, Validation Accuracy: {correct / total:.6f}')

        if val_loss < minimum_val_loss:
            minimum_val_loss = val_loss
            epochs_no_improve = 0
            save_model(model, 'best_model.pth')

        else:
            epochs_no_improve += 1
            if epochs_no_improve == 5:
                print('Early stopping!')
                break


def sentence2vec(sentence):
    w2v_size = wv.vector_size
    words = sentence.split()
    sentence_vector = np.mean([wv[word] for word in words if word in wv] or [np.zeros(w2v_size)], axis=0)
    return sentence_vector

def load_to_loader(file_path, test=False):
    data = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)
    df['postTextVec'] = df['postText'].apply(lambda x: sentence2vec(x[0]))
    df['targetParagraphsVec'] = df['targetParagraphs'].apply(lambda x: np.mean([sentence2vec(i) for i in x], axis=0))
    df['targetLengths'] = df['targetParagraphs'].apply(lambda x: len(x))
    df['textLengths'] = df['postText'].apply(lambda x: len(x[0]))
    df['feature'] = df.apply(lambda x: np.concatenate([x['postTextVec'], x['targetParagraphsVec']]), axis=1)
    # Convert to train_loader
    train_x = np.array(df['feature'].tolist())

    train_x = torch.tensor(train_x, dtype=torch.float32).to(device)
    if test:
        return train_x
    df['tagsS'] = df['tags'].apply(lambda x: {'multi': 0, 'passage': 1, 'phrase': 2}[x[0]])
    train_y = np.array(df['tagsS'].tolist())
    train_y = torch.tensor(train_y, dtype=torch.long).to(device)
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)
    return train_loader


train_loader = load_to_loader(
    r"train.jsonl")
val_loader = load_to_loader(
    r"val.jsonl")
test = load_to_loader(
    r"test.jsonl",
    test=True)

for model in ['sigmoid']:
    print(f'Training model with activation function: {model}')
    tc_model = TextClassifier(model).to(device)


    train_model(tc_model, train_loader, val_loader)
    # Evaluate on test set
    tc_model.eval()
    save_model(tc_model, f'{model}.model')

tc_model = load_model(tc_model, 'best_model.pth')
outputs = tc_model(test)
_, predicted = torch.max(outputs.data, 1)

backdict={0:'multi', 1:'passage', 2: 'phrase'}

with open("solution.csv", "w") as f:
    f.write("id,spoilerType\n")
    for ind,pre in enumerate(predicted):
        f.write(f"{ind},{backdict[pre.item()]}\n")

assert 1==2
