import torch
from torch import nn
import torch.nn.functional as F
import random

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
    def __init__(self, cnn_dims, dropout):
        super(coreAlgo, self).__init__()
        self.cnn_dims = [1] + cnn_dims + [1]
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
    def forward(self, x):
        for c in self.cnn:
            x = c(x)
        return x
class Encoder(nn.Module):
    def __init__(self, num_inputs, cnn_dims, dropout):
        super(Encoder, self).__init__()
        self.preproc = preProc(num_inputs)
        self.core = coreAlgo(cnn_dims, dropout)
        
    def forward(self, s):
        X, Y, c = self.preproc(s)
        X = self.core(X)
        Y = self.core(Y)
        
        return X.squeeze(1), Y.squeeze(1)
class Attention(nn.Module):
    def __init__(self, num_inputs, cnn_dims, dropout):
        super(Attention, self).__init__()
        self.core = coreAlgo(cnn_dims, dropout)
    def forward(self, x0, x1):
        x0, x1 = [x.unsqueeze(1) for x in [x0, x1]]
        x0 = self.core(x0)
        x1 = self.core(x1)
        x0, x1 = [x.squeeze(1) for x in [x0, x1]]
        x = x1 @ x0.T
        return F.softmax(x.mean(1), dim=-1).unsqueeze(1)
class Decoder(nn.Module):
    def __init__(self, num_inputs, attention, dropout):
        super(Decoder, self).__init__()
        self.attention = attention
        self.cat = nn.Linear(num_inputs, 64)
    def forward(self, X, Y, X_, Y_):
        X_aw = self.attention(X_, X)
        Y_aw = self.attention(Y_, Y)
        X_cat = torch.cat((X, X_aw), 1)
        Y_cat = torch.cat((Y, Y_aw), 1)
        
        X_cat = F.relu(self.cat(X_cat))
        Y_cat = F.relu(self.cat(Y_cat))
        S = X_cat @ Y_cat.T
        action_scores = S.mean(0)
        return action_scores, X, Y
class Policy(nn.Module):
    def __init__(self, num_inputs, cnn_dims, dropout=0):
        super(Policy, self).__init__()
        self.encoder = Encoder(
                        num_inputs, cnn_dims, dropout
                        )
        self.attention = Attention(
                        num_inputs, cnn_dims, dropout
                        )
        self.decoder = Decoder(
                        num_inputs+2 - (len(cnn_dims) + 1) * 2, self.attention, dropout
                        )
        self.num_inputs = num_inputs
        self.cnn_dims = cnn_dims
    def forward(self, s, prev=None):
        if prev:
            X_, Y_ = prev
        else:
            X_ = torch.rand(self.num_inputs, self.num_inputs + 1 - (len(self.cnn_dims) + 1) * 2)
            Y_ = torch.rand(self.num_inputs, self.num_inputs + 1 - (len(self.cnn_dims) + 1) * 2)
        X, Y = self.encoder(s)
        action_scores, X, Y = self.decoder(X, Y, X_, Y_)
        prob = F.softmax(action_scores, dim=-1)
        return prob, (X, Y)
