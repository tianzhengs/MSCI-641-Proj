import numpy as np
import pandas as pd
import json
import gensim.downloader as api
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch import save

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# Load pre-trained Word2Vec model
wv = api.load('word2vec-google-news-300')


def save_model(model, filepath):
    save(model.state_dict(), filepath)


def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set the model to evaluation mode
    return model

class CNNModel(nn.Module):
    def __init__(self, activation_fn='ReLU', input_channels=1, num_classes=3, dropout_rate=0.5):
        super(CNNModel, self).__init__()

        if activation_fn == 'ReLU':
            self.activation = nn.ReLU()
        elif activation_fn == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_fn == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('Invalid activation function')

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=300, out_channels=300, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=300, out_channels=300, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout_rate)

        self.conv3 = nn.Conv1d(in_channels=300, out_channels=300, kernel_size=3, padding=1)


        # Fully connected layers
        self.fc1 = nn.Linear(300*23, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change the shape to (batch_size, 300, 92)
        x = self.activation(self.conv1(x))  # (batch_size, 300, 92)
        x = self.activation(self.conv2(x))  # (batch_size, 300, 92)
        x = self.pool(x)  # (batch_size, 300, 46)
        x = self.pool(self.activation(self.conv3(x)))  # (batch_size, 300, 23)
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 300*23)
        x = self.dropout(x)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(model, train_loader, val_loader, num_epochs=1000):
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
            if epochs_no_improve == 30:
                print('Early stopping!')
                break


def sentence2vec(sentence):
    words = sentence.split()
    sentence_vector = np.array([wv[word] if word in wv else np.zeros(wv.vector_size) for word in words])
    # sentence_vector size: 93*wv.vector_size
    # padding to this size (for sentences with less than 93 words)
    sentence_vector = np.pad(sentence_vector, ((0, 93 - len(sentence_vector)), (0, 0)), 'constant')
    # clipping to this size (for sentences with more than 93 words)
    sentence_vector = sentence_vector[:93]
    return sentence_vector


def load_to_loader(file_path, test=False):
    data = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)
    df['feature'] = df['postText'].apply(lambda x: sentence2vec(x[0]))

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


train_loader = load_to_loader("train.jsonl")
val_loader = load_to_loader("val.jsonl")
test = load_to_loader("test.jsonl", test=True)

# Training the model
for model in ['ReLU']:
    print(f'Training model with activation function: {model}')
    tc_model = CNNModel(model)
    tc_model = tc_model.to(device)
    train_model(tc_model, train_loader, val_loader)

tc_model = load_model(tc_model, 'best_model.pth')

with torch.no_grad():
    outputs = tc_model(test)
    _, predicted = torch.max(outputs.data, 1)
    backdict = {0: 'multi', 1: 'passage', 2: 'phrase'}

    with open("solution.csv", "w") as f:
        f.write("id,spoilerType\n")
        for ind, pre in enumerate(predicted):
            f.write(f"{ind},{backdict[pre.item()]}\n")
