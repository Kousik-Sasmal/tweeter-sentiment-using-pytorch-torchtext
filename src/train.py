
import config
from dataset import CustomDataset
from model import RNNClassify, LSTMClassify,LSTM2Classify
from engine import test_step,train_step
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
import pickle

import os
import logging
import warnings
warnings.filterwarnings("ignore")

# Due to warning when initializing the "spacy" tokenizer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable tensorflow logging
logging.getLogger('tensorflow').disabled = True  # disable tensorflow warning messages



df = pd.read_csv('artifacts/train_cleaned.csv')
df_valid = pd.read_csv('artifacts/valid_cleaned.csv')

# limit df
# df = df[:10000]
# df_valid = df_valid[:1000]


# creating vocabulary   
tokenizer = get_tokenizer("spacy")

def yield_tokens(data_iter):
    for text in data_iter:
        text = text.lower()
        
        yield tokenizer(text)
token_generator = yield_tokens(df['tweet'])
        
vocab = build_vocab_from_iterator(token_generator, specials=["<UNK>"],max_tokens=config.VOCAB_SIZE)
vocab.set_default_index(vocab["<UNK>"])


# Creating collate function that will transform the data of a batch
def collate_fn(samples):
    # Separate the texts and targets from the samples in a batch
    texts, targets = zip(*samples)
    
    tokenized_texts = [tokenizer(text.lower()) for text in texts]
    text_indices = [torch.tensor(vocab(token)) for token in tokenized_texts]
    
    # Pad the text sequences to have the same length
    padded_texts = torch.nn.utils.rnn.pad_sequence(text_indices, batch_first=True)
    
    target_tensor = torch.tensor(targets)
    
    return padded_texts, target_tensor



train_data = CustomDataset(df['tweet'],df['label'])
test_data = CustomDataset(df_valid['tweet'],df_valid['label'])


train_dataloader = DataLoader(train_data,batch_size=20,collate_fn=collate_fn)
test_dataloader = DataLoader(test_data,batch_size=20,collate_fn=collate_fn)


def train(model,loss_fn,optimizer):

    epochs  = config.EPOCHS

    best_accuracy = 0.0
    best_epoch = 0
    for epoch in tqdm(range(epochs)):
        train_loss, train_accuracy = train_step(model,train_dataloader,loss_fn,optimizer)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        
        test_loss, test_accuracy = test_step(model,test_dataloader,loss_fn)
        print(f'Epoch {epoch+1}/{epochs}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        print('--'*25)    
    
        # Check if current test_accuracy is better than the best_accuracy
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
            
            # Save the model
            torch.save(model.state_dict(), 'artifacts/best_model.pth')


    print(f"Best accuracy: {best_accuracy} at epoch: {best_epoch}")



if __name__ == "__main__":
    pickle.dump(vocab,open('artifacts/vocabulary.pkl','wb'))

    model = LSTM2Classify().to(config.DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.LEARNING_RATE)
    
    loss_fn = torch.nn.CrossEntropyLoss()  # remember it gives logits (row outputs)
    
    train(model=model,loss_fn=loss_fn,optimizer=optimizer)

        
