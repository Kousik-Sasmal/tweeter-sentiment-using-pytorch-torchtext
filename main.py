import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import numericalize_tokens_from_iterator
from torch.nn.utils.rnn import pad_sequence

import logging
import os
import warnings
warnings.filterwarnings("ignore")

# Due to warning when initializing the "spacy" tokenizer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable tensorflow logging
logging.getLogger('tensorflow').disabled = True  # disable tensorflow warning messages

device = "cuda" if torch.cuda.is_available() else "cpu"

# train data
df = pd.read_csv('data/train_cleaned.csv')
# limit df to 1000
df = df[:1000]

# test data
df_valid = pd.read_csv('data/valid_cleaned.csv')
# limit df to 100
df_valid = df_valid[:100]


tokenizer = get_tokenizer("spacy")

def token_gen(text):
    """
    Tokenizes each sentence in a given text and yields the resulting tokens.

    Args:
        text (list[str]): A list of sentences to tokenize.

    Yields:
        list[str]: The resulting tokens from each sentence.
    """
    for sent in text:
        tokens = tokenizer(sent)
        yield tokens

vocab = build_vocab_from_iterator(token_gen(df['tweet']),specials=["<UNK>"])
#print(vocab.get_stoi())
vocab.set_default_index(vocab["<UNK>"])  ## to handel OOV problem


# numericalize tokens from iterator using vocab
sequence = numericalize_tokens_from_iterator(vocab=vocab.get_stoi(),iterator=token_gen(df['tweet']))

# for ids in sequence:
#     print([num for num in ids])


# create a list to store tokenized sequences
text = []
for i in range(len(df)):
    x = list(next(sequence))
    text.append(x)

# Pad the sequences to the same length along dimension 0
padded_text = pad_sequence([torch.tensor(x) for x in text], batch_first=True, padding_value=0)

# label of the sentiment for train data
label =df['label'].to_list()
label= torch.tensor(label)

# X_train,y_train
X_train,y_train = padded_text,label



valid_token_ids = []
for i in range(len(df_valid)):
    token_id = vocab(tokenizer(df_valid['tweet'][i]))
    valid_token_ids.append(token_id)
    

# Pad the sequences to the same length along dimension 0
padded_text_valid = pad_sequence([torch.tensor(x) for x in valid_token_ids], batch_first=True, padding_value=0)
# here look, <UNK> will be assign to 0 and padding_idx will be assign also 0

label_valid = df_valid['label'].to_list()
label_valid = torch.tensor(label_valid)

# X_test,y_test
X_test, y_test = padded_text_valid, label_valid


# Determine the number of classes
num_classes = len(label.unique())

# Define the RNNClassify module
class RNNClassify(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        
        # Define the embedding layer
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Define the RNN layer
        self.rnn = nn.RNN(embed_dim, hidden_size,batch_first=True)
        
        # Define the linear layer
        self.linear = nn.Linear(hidden_size, num_classes)
        
        # Initialize the weights of the module
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.5
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.rnn.weight_ih_l0.data.uniform_(-initrange, initrange)
        self.rnn.weight_hh_l0.data.uniform_(-initrange, initrange)
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        
    def forward(self, input):
        # Embed the input
        embedded = self.embed(input)
        #print('embedded shape:',embedded.shape)
        
        # Pass the embedded input through the RNN layer
        output, hidden = self.rnn(embedded)
        #print('rnn output shape:',output.shape)
        #print('rnn hidden shape:',hidden.shape)
        
        output = output[:, -1, :]  # taking last output of RNN
        #print('rnn last output shape:',output.shape)
        
        # Pass the output through the linear layer
        output = self.linear(output)
        
        # Return the output
        return output
    


VOCAB_SIZE = len(vocab.get_stoi())
model = RNNClassify(vocab_size=VOCAB_SIZE,embed_dim=100,hidden_size=32).to(device)



optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()  # remember it gives logits (row outputs)


# Batch Gradient Descent
epochs = 60

for epoch in range(epochs):
    train_loss,train_acc = 0,0

    # Set model to training mode
    model.train()

    X_train,y_train = X_train.to(device), y_train.to(device)

    y_logits = model(X_train)

    loss = loss_fn(y_logits, y_train)

    train_loss += loss
    train_acc += (y_logits.argmax(1) == y_train).sum().item() / len(y_train)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')

    # Set model to eval mode
    model.eval()
    with torch.inference_mode():
        
        test_loss,test_acc = 0,0

        X_test, y_test = X_test.to(device), y_test.to(device)
        
        y_logits = model(X_test)

        # Compute loss with one-hot encoded targets
        loss = loss_fn(y_logits, y_test)

        test_loss += loss.item()
            
        # Compute accuracy
        test_preds = y_logits.argmax(dim=1)
        test_acc += (test_preds == y_test).sum().item() / len(y_test)

        
        print(f'Epoch {epoch+1}/{epochs}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
        
        print('--'*50)