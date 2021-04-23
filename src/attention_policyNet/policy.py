import torch
from torch import nn
import torch.nn.functional as F
import random
from torch.nn.utils.rnn import pad_sequence

class preProc(nn.Module):
    def __init__(self, num_inputs):
        super(preProc, self).__init__()
        self.batchNormMatrix = nn.BatchNorm1d(num_features = num_inputs)
        
    def forward(self, s):
        a, b, c, d, e = [torch.FloatTensor(item) for item in s]
        a, d = [self.batchNormMatrix(x) for x in [a, d]]
        b, e =[(x - x.min()) / (x.max() - x.min()) for x in [b, e]]
        X, Y = [torch.cat((x, y.unsqueeze(1)), 1) for x, y in zip([a, d], [b, e])]
        X, Y = [x.unsqueeze(1) for x in [X, Y]]
        return X, Y, c.long()
class coreAlgo(nn.Module):
    def __init__(self, num_inputs , cnn_dims, rnn_hidden, dropout):
        super(coreAlgo, self).__init__()
        self.cnn_dims = [1] + cnn_dims
        self.cnn = nn.ModuleList(
                    nn.Sequential(
                            nn.Conv1d(x, y, 3),
                            nn.Dropout(dropout),
                            nn.ReLU()
                        ) for x, y in zip(
                        self.cnn_dims[:-1],
                        self.cnn_dims[1:]
                        )
                    )
        self.rnn = nn.GRU(num_inputs+1 - (len(self.cnn_dims)-1) * 2, rnn_hidden)
        self.out = nn.Linear(rnn_hidden, 1)
        
    def forward(self, x, hidden=None):
        for c in self.cnn:
            x = c(x)
        if hidden!=None:
            x, x_h = self.rnn(x, hidden)
        else:
            x, x_h = self.rnn(x)
            
        x = F.relu(self.out(x))
        return x.squeeze(), x_h
class Encoder(nn.Module):
    def __init__(self, num_inputs , cnn_dims, rnn_hidden, dropout):
        super(Encoder, self).__init__()
        self.preproc = preProc(num_inputs)
        self.core = coreAlgo(num_inputs , cnn_dims, rnn_hidden, dropout)
        
    def forward(self, s):
        X, Y, c = self.preproc(s)
        X, _ = self.core(X)
        Y, _ = self.core(Y)
        
        return X, Y
class Attention(nn.Module):
    def __init__(self, input_dim, linears, dropout):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.linears = [input_dim] + linears
        self.fcs = nn.ModuleList(
                    nn.Sequential(
                            nn.Linear(x, y),
                            nn.ReLU()
                        ) for x, y in zip(
                        self.linears[:-1],
                        self.linears[1:]
                        )
                    )
    def forward(self, x0, x1):
        for fc in self.fcs:
            x0 = fc(x0)
            x1 = fc(x1)
        x = x0 @ x1.T
        return F.normalize(torch.tanh(x.mean(1)).unsqueeze(1))
class Decoder(nn.Module):
    def __init__(self, num_inputs, cnn_dims, rnn_hidden, dropout):
        super(Decoder, self).__init__()
        
        self.preproc = preProc(num_inputs)
        
        self.attention = Attention(cnn_dims[-1], cnn_dims, dropout)
        self.core = coreAlgo(num_inputs , cnn_dims, rnn_hidden, dropout)
        self.cat = nn.Linear(cnn_dims[-1] + 1, 64)
        
    
    def forward(self, s, XYs, hidden):
        X, Y, _ = self.preproc(s)
        Xs, Ys = XYs
        xh, yh = hidden
        
        X, xh = self.core(X, xh)
        Y, yh = self.core(Y, yh)
        if Xs == None and Ys == None:
            X_aw = torch.zeros(X.shape[0], 1)
            Y_aw = torch.zeros(Y.shape[0], 1)
        else:
            X_aw = self.attention(X, Xs)
            Y_aw = self.attention(Y, Ys)
        X_cat = torch.cat((X, X_aw), 1)
        Y_cat = torch.cat((Y, Y_aw), 1)
        
        X_cat = F.relu(self.cat(X_cat))
        Y_cat = F.relu(self.cat(Y_cat))
        
        S = X_cat @ Y_cat.T
        action_scores = S.mean(0)
        return action_scores, xh, yh
class Policy(nn.Module):
    def __init__(self, num_inputs , cnn_dims, rnn_hidden, dropout=0):
        super(Policy, self).__init__()
        self.encoder = Encoder(
                        num_inputs , cnn_dims, rnn_hidden, dropout
                        )
        self.decoder = Decoder(
                        num_inputs, cnn_dims, rnn_hidden, dropout
                        )
        self.num_inputs = num_inputs
        self.hiddens = (
                        torch.rand(1, cnn_dims[-1], rnn_hidden),
                        torch.rand(1, cnn_dims[-1], rnn_hidden)
                        )
    def forward(self, s, obss):
        if obss == []:
            Xs, Ys = None, None
        else:
            Xs = torch.cat([self.encoder(s)[0] for s in obss])
            Ys = torch.cat([self.encoder(s)[1] for s in obss])
        action_scores, X_h, Y_h = self.decoder(s, (Xs, Ys), self.hiddens)
        self.hiddens = (X_h.detach(), Y_h.detach())
        prob = F.softmax(action_scores, dim=-1)
        return prob