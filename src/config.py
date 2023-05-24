import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


OUTPUT_CLASS = 4

EMBED_DIM = 128

HIDDEN_SIZE = 32

VOCAB_SIZE = 20000

BATCH_SIZE = 20

EPOCHS = 20

LEARNING_RATE = 0.001

LSTM_DROPOUT = 0.2

LAYER_DROPOUT = 0.1
