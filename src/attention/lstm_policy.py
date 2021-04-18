import torch
from torch import nn
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_layers, dropout=0):
        super(Encoder, self).__init__()
        self.batchNormMatrix = nn.BatchNorm1d(num_features = num_inputs)
        self.lstm = nn.LSTM(
                    num_inputs+1,
                    hidden_size,
                    num_layers,
                    bidirectional=False,
                    dropout = dropout
                    )
    def forward(self, s):
        a, b, _, d, e = self._preproc(s)
        a, d = [self.batchNormMatrix(x) for x in [a, d]]
        b, e =[(x - x.min()) / (x.max() - x.min()) for x in [b, e]]
        X, Y = [torch.cat((x, y.unsqueeze(1)), 1) for x, y in zip([a, d], [b, e])]
        X, Y = [x.unsqueeze(1) for x in [X, Y]]
        
        X, (X_h, X_c) = self.lstm(X)
        Y, (Y_h, Y_c) = self.lstm(Y)
        
        return (X, X_h, X_c), (Y, Y_h, Y_c)
    def _preproc(self, s):
        return [torch.FloatTensor(item) for item in s]

class Attention(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(torch.mul(hidden, encoder_output), dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(torch.mul(hidden, energy), dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(torch.mul(self.v, energy), dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
class Decoder(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_layers, attention, dropout=0):
        super(Decoder, self).__init__()
        self.batchNormMatrix = nn.BatchNorm1d(num_features = num_inputs)
        self.lstm = nn.LSTM(
                    num_inputs+1,
                    hidden_size,
                    num_layers,
                    bidirectional=False,
                    dropout = dropout
                    )
        self.attention = attention
        self.cat = nn.Linear(hidden_size *2, hidden_size)
        
        
    def forward(self, s, X_hidden, Y_hidden ,X_, Y_):
        a, b, _, d, e = self._preproc(s)
        a, d = [self.batchNormMatrix(x) for x in [a, d]]
        b, e =[(x - x.min()) / (x.max() - x.min()) for x in [b, e]]
        X, Y = [torch.cat((x, y.unsqueeze(1)), 1) for x, y in zip([a, d], [b, e])]
        X, Y = [x.unsqueeze(1) for x in [X, Y]]
        
        X, (X_h, X_c) = self.lstm(X, X_hidden)
        Y, (Y_h, Y_c) = self.lstm(Y, Y_hidden)
        
        
        X_, _, _ = X_
        Y_, _, _ = Y_
        
        X_attn_weights, Y_attn_weights = [self.attention(x, x_) for x, x_ in zip((X, Y), (X_, Y_))]
        
        X, Y = X.squeeze(1), Y.squeeze(1)
        X_context = X_attn_weights.bmm(X_).squeeze(1)
        Y_context = Y_attn_weights.bmm(Y_).squeeze(1)
        
        
        X_cat_input = torch.cat((X, X_context), 1)
        X_cat_output = F.relu(self.cat(X_cat_input))
        
        Y_cat_input = torch.cat((Y, Y_context), 1)
        Y_cat_output = F.relu(self.cat(Y_cat_input))
        S = X_cat_output @ Y_cat_output.T
        
        action_scores = S.mean(0)
        
        
        return F.softmax(action_scores, dim=-1), (X_h, X_c), (Y_h, Y_c)
    
    def _preproc(self, s):
        return [torch.FloatTensor(item) for item in s]
class Policy(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_layers, method, 
                epsilon=.8, epsilon_decay = .8, dropout=0):
        super(Policy, self).__init__()
        self.encoder = Encoder(
                        num_inputs, 
                        hidden_size, 
                        num_layers, 
                        dropout
                        )
        attention = Attention(
                        method,
                        hidden_size
                        )
        self.decoder = Decoder(
                        num_inputs, 
                        hidden_size, 
                        num_layers, 
                        attention, 
                        dropout
                        )
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
    def forward(self, s, hidden=None):
        if hidden:
            X_hidden, Y_hidden = hidden
        else:
            X_hidden = torch.zeros(self.num_layers, 1, self.hidden_size), \
                        torch.zeros(self.num_layers, 1, self.hidden_size)
            Y_hidden = torch.zeros(self.num_layers, 1, self.hidden_size), \
                        torch.zeros(self.num_layers, 1, self.hidden_size) 
        X, Y = self.encoder(s)
        prob, (X_h, X_c), (Y_h, Y_c) = self.decoder(s, X_hidden, Y_hidden, X, Y)
        if random.random() < self.epsilon:
            prob = torch.rand(prob.shape)
            prob /= prob.sum()
        return prob, ((X_h, X_c), (Y_h, Y_c))