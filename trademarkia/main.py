import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torchtext import Field, TabularDataset, BucketIterator
import torch
import torch.nn as nn
import torch.optim as optim


class SolutionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_length, num_classes):
        super(SolutionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1d = nn.Conv1d(embedding_dim, 128, kernel_size=5)
        self.maxpool1d = nn.MaxPool1d(kernel_size=3)
        self.lstm1 = nn.LSTM(128, 256, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(512, 128, bidirectional=True, batch_first=True)
        self.dense1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, num_classes)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        conv_out = self.conv1d(embeddings.transpose(1, 2))
        pool_out = self.maxpool1d(conv_out).squeeze(2)
        lstm1_out, _ = self.lstm1(pool_out)
        lstm2_out, _ = self.lstm2(lstm1_out)
        dense1_out = self.dense1(lstm2_out[:, -1, :])
        dropout_out = self.dropout(dense1_out)
        dense2_out = self.dense2(dropout_out)
        output = self.dense3(dense2_out)
        return output


def solution_model():
    vocab_size = 10000
    embedding_dim = 30
    max_length = 40
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 50000
    num_classes = 45  # Update the number of classes
    num_epochs = 30
    batch_size = 256

    with open("shuffled_file.json", 'r') as f:
        datastore = json.load(f)

    sentences = [item['description'] for item in datastore]
    labels = [item['class_id'] for item in datastore]

    training_sentences = sentences[:training_size]
    testing_sentences = sentences[training_size:]
    training_labels = labels[:training_size]
    testing_labels = labels[training_size:]

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index

    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    # Convert labels to numerical values
    label_encoder = LabelEncoder()
    training_labels_encoded = label_encoder.fit_transform(training_labels)
    testing_labels_encoded = label_encoder.transform(testing_labels)

    # Create PyTorch datasets and data loaders
    fields = [('description', Field(sequential=True)), ('class_id', Field(sequential=False))]
    train_examples = [({'description': description, 'class_id': class_id}) for description, class_id in
                      zip(training_padded, training_labels_encoded)]
    test_examples = [({'description': description, 'class_id': class_id}) for description, class_id in
                     zip(testing_padded, testing_labels_encoded)]
    train_data = TabularDataset(train_examples, format='json', fields=fields)
    test_data = TabularDataset(test_examples, format='json', fields=fields)
    train_loader, test_loader = BucketIterator.splits((train_data, test_data), batch_size=batch_size, shuffle=True,
                                                      sort=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SolutionModel(vocab_size, embedding_dim, max_length, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0

        for batch in train_loader:
            inputs = batch.description.to(device)
            targets = batch.class_id.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            accuracy = (outputs.argmax(dim=1) == targets).float().mean()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()

        epoch_loss /= len(train_loader)
        epoch_accuracy /= len(train_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    # Save the model
    torch.save(model.state_dict(), "trademarker.pth")

    # Save the tokenizer as a JSON file
    tokenizer_json = tokenizer.to_json()
    with open('tokenizer.json', 'w') as f:
        json.dump(tokenizer_json, f)

    # Save the label encoder as a JSON file
    label_encoder_json = json.dumps(list(label_encoder.classes_))
    with open('label_encoder.json', 'w') as f:
        f.write(label_encoder_json)

    return model


if __name__ == '__main__':
    model = solution_model()



