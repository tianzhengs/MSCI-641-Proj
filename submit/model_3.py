import numpy as np
import pandas as pd
import json
import gensim.downloader as api
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch import save
from nltk.stem.wordnet import WordNetLemmatizer
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

lmtzr = WordNetLemmatizer()

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


class LSTMModel(nn.Module):
    def __init__(self, input_size=300, hidden_size=256, num_layers=2, num_classes=3, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # * 2 for bidirectional

    def forward(self, x, lengths):
        # Pack the padded sequence
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)

        # LSTM forward pass
        packed_output, _ = self.lstm(packed_input)

        # Unpack the sequence
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Get the output of the last non-padded element in each sequence
        last_output = output[torch.arange(output.size(0)), lengths - 1]

        # Dropout and fully connected layer
        out = self.dropout(last_output)
        out = self.fc(out)
        return out


def train_model(model, train_loader, val_loader, num_epochs=1000):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # L2 regularization
    minimum_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        for inputs, lengths, labels in train_loader:
            inputs, lengths, labels = inputs.to(device), lengths.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, lengths, labels in val_loader:
                inputs, lengths, labels = inputs.to(device), lengths.to(device), labels.to(device)
                outputs = model(inputs, lengths)
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
            save_model(model, '../best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve == 30:
                print('Early stopping!')
                break


def sentence2vec(sentence):
    words = sentence.split()
    sentence_vector = [wv[lmtzr.lemmatize(word)] if lmtzr.lemmatize(word) in wv else np.zeros(wv.vector_size) for word
                       in words]
    return sentence_vector


def load_to_loader(file_path, test=False):
    data = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)
    df['feature'] = df['postText'].apply(lambda x: sentence2vec(x[0]))

    # Convert to tensors
    features = [torch.tensor(seq, dtype=torch.float32) for seq in df['feature']]

    # Get sequence lengths
    lengths = torch.tensor([len(seq) for seq in features])

    # Sort sequences by descending length
    lengths, sort_idx = lengths.sort(descending=True)
    features = [features[i] for i in sort_idx]

    # Pad sequences
    features_padded = pad_sequence(features, batch_first=True)

    if test:
        return features_padded.to(device), lengths.to(device)

    df['tagsS'] = df['tags'].apply(lambda x: {'multi': 0, 'passage': 1, 'phrase': 2}[x[0]])
    labels = torch.tensor(df['tagsS'].tolist(), dtype=torch.long)[sort_idx].to(device)

    dataset = TensorDataset(features_padded, lengths, labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    return loader


train_loader = load_to_loader("../train.jsonl")
val_loader = load_to_loader("../val.jsonl")
test, test_lengths = load_to_loader("../test.jsonl", test=True)

# Training the model
print(f'Training model')
tc_model = LSTMModel()
tc_model = tc_model.to(device)
train_model(tc_model, train_loader, val_loader)

tc_model = load_model(tc_model, '../best_model.pth')

with torch.no_grad():
    outputs = tc_model(test, test_lengths)
    _, predicted = torch.max(outputs.data, 1)
    backdict = {0: 'multi', 1: 'passage', 2: 'phrase'}

    with open("../solution.csv", "w") as f:
        f.write("id,spoilerType\n")
        for ind, pre in enumerate(predicted):
            f.write(f"{ind},{backdict[pre.item()]}\n")

print("Done!")