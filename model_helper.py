import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.datasets import text_classification
from torchtext.data.utils import get_tokenizer, ngrams_iterator
from torchtext.data.utils import get_tokenizer

# Set device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define preprocessing pipeline
tokenizer = get_tokenizer("basic_english")
NGRAMS = 2

def text_pipeline(text):
    tokens = tokenizer(text)
    return list(ngrams_iterator(tokens, NGRAMS))

# Download and prepare the data
train_dataset, test_dataset = text_classification.DATASETS["AG_NEWS"](
    root="./data", ngrams=NGRAMS, vocab=None, include_unk=False
)

# Set up the data iterators
BATCH_SIZE = 16
train_iter = torchtext.data.BucketIterator(
    train_dataset, batch_size=BATCH_SIZE, device=device, shuffle=True
)

test_iter = torchtext.data.BucketIterator(
    test_dataset, batch_size=BATCH_SIZE, device=device, shuffle=True
)

# Define the model
class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

# Set up the model and optimizer
VOCAB_SIZE = len(train_dataset.get_vocab())
EMBED_DIM = 32
NUM_CLASS = len(train_dataset.get_labels())
model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUM_CLASS).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Train the model
def train(model, train_iter, optimizer, criterion):
    model.train()
    train_loss = 0
    train_acc = 0
    for batch in train_iter:
        optimizer.zero_grad()
        text, offsets = batch.text
        labels = batch.label
        output = model(text, offsets)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += (output.argmax(1) == labels).sum().item()
    train_loss /= len(train_iter)
    train_acc /= len(train_iter.dataset)
    return train_loss, train_acc

# Evaluate the model
def evaluate(model, test_iter, criterion):
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for batch in test_iter:
            text, offsets = batch.text
            labels = batch.label
            output = model(text, offsets)
            loss = criterion(output, labels)
            test_loss += loss.item()
            test_acc += (output.argmax(1) == labels).sum().item()
    test_loss /= len(test_iter)
    test_acc /= len(test_iter.dataset)
    return test_loss, test_acc
