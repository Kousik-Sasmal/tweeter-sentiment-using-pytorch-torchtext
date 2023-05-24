
import config
import torch.nn as nn

class RNNClassify(nn.Module):
    def __init__(self, vocab_size=config.VOCAB_SIZE, embed_dim=config.EMBED_DIM, 
                  hidden_size=config.HIDDEN_SIZE,output_classes=config.OUTPUT_CLASS):
        super().__init__()
        
        # Define the embedding layer
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.rnn = nn.RNN(embed_dim, hidden_size,batch_first=True)   
        self.linear = nn.Linear(hidden_size, output_classes)
        
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

        output = output[:, -1, :]  # taking last output of RNN
        #print('rnn last output shape:',output.shape)
        
        # Pass the output through the linear layer
        output = self.linear(output)
        
        return output




class LSTMClassify(nn.Module):
    
    def __init__(self, vocab_size=config.VOCAB_SIZE, embed_dim=config.EMBED_DIM, 
                  hidden_size=config.HIDDEN_SIZE,output_classes=config.OUTPUT_CLASS):
        super().__init__()
        
        # Define the embedding layer
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(embed_dim, hidden_size,batch_first=True)
        
        self.linear = nn.Linear(hidden_size, output_classes)

        
    def forward(self, input):
        embedded = self.embed(input)        
        # Pass the embedded input through the LSTM layer
        output, (hidden,cell) = self.lstm(embedded)
        
        
        output = output[:, -1, :] 
        #print('LSTM last output shape:',output.shape)
        
        # Pass the output through the linear layer
        output = self.linear(output)
        
        return output





class LSTM2Classify(nn.Module):
    def __init__(self, vocab_size=config.VOCAB_SIZE, embed_dim=config.EMBED_DIM, hidden_size=config.HIDDEN_SIZE,
                  output_classes=config.OUTPUT_CLASS, num_layers=2,bidirectional=True, lstm_dropout=config.LSTM_DROPOUT, 
                  dropout=config.LAYER_DROPOUT):
        super().__init__()
        
        # Define the embedding layer
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers, batch_first=True, 
                            bidirectional=bidirectional, dropout=lstm_dropout)
        
        lstm_output_size = hidden_size
        if bidirectional:
            lstm_output_size *= 2
        
        self.linear = nn.Linear(lstm_output_size, output_classes)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, input):
        # Embed the input
        embedded = self.embed(input)
        
        # Pass the embedded input through the LSTM layer
        output, (_, _) = self.lstm(embedded)
        
        output = output[:, -1, :]  # taking the last output of the LSTM
        
        # dropout layer
        output = self.dropout(output)

        # Pass the output through the linear layer
        output = self.linear(output)
        
        # Return the output
        return output

