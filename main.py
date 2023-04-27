import numpy as np
import pandas as pd
import torchtext
import zipfile
import pathlib
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import logging
import os
import warnings
warnings.filterwarnings("ignore")



# Due to warning when initializing the "spacy" tokenizer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable tensorflow logging
logging.getLogger('tensorflow').disabled = True  # disable tensorflow warning messages




df = pd.read_csv('data/train_cleaned.csv')
df.fillna("",inplace=True)
df = df[:1000]

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

vocab = build_vocab_from_iterator(token_gen(df['tweet']),specials=["<unk>"],special_first=False)

#print(vocab.get_stoi())

from torchtext.data.functional import numericalize_tokens_from_iterator
sequence = numericalize_tokens_from_iterator(vocab=vocab.get_stoi(),iterator=token_gen(df['tweet']))

for ids in sequence:
    print([num for num in ids])