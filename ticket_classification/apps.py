import os
from django.apps import AppConfig
from django.conf import settings

import torch
import torch.nn as nn
import pickle

    
with open(os.path.join(settings.MODELS, "encoder.pickle"), 'rb') as handle:
    le = pickle.load(handle)
    
with open(os.path.join(settings.MODELS, "fasttext.pickle"), 'rb') as handle:
    embedding_matrix = pickle.load(handle)
    
    
embed_size = 300
batch_size = 32
max_features = 12000
hidden_size = 256
n_layers = 1
drop_prob = 0.4
n_classes = len(le.classes_)
    
class RNNLSTM(nn.Module):
    
    def __init__(self):
        super(RNNLSTM, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        
        # RNN Layer
        self.rnn= nn.LSTM(embed_size, hidden_size, n_layers, bidirectional= True, dropout=0.1)
        
        # hidden layer linear transformation
        self.fc = nn.Linear(hidden_size*4 , 64)   
        # activation function
        self.relu = nn.ReLU()
        
        # Droupout layer
        self.dropout = nn.Dropout(drop_prob)
        
        # Output layer
        self.out = nn.Linear(64, n_classes)


    def forward(self, x):
        out_embedding = self.embedding(x)
        out_rnn, _ = self.rnn(out_embedding)
        # pooling operation
        avg_pool = torch.mean(out_rnn, 1)
        max_pool,_ = torch.max(out_rnn, 1)
        conc = torch.cat((avg_pool, max_pool), 1)
        
        conc = self.relu(self.fc(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        
        # return final output
        return out


class TicketClassificationConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ticket_classification'
    model = RNNLSTM()
    model.load_state_dict(torch.load(os.path.join(settings.MODELS, "lstmmodel.pt")))