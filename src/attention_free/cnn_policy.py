import torch
from torch import nn
import torch.nn.functional as F
class Policy(nn.Module):
    def __init__(self, num_inputs):
        super(Policy, self).__init__()
        self.batchNormMatrix = nn.BatchNorm1d(num_features = num_inputs)
        self.core = nn.Sequential(
                                nn.Conv1d(1, 32, 3),
                                nn.Dropout(.5),
                                nn.ReLU(),
                                nn.Conv1d(32, 64, 3),
                                nn.Dropout(.5),
                                nn.ReLU(),
                                nn.Conv1d(64, 32, 3),
                                nn.Dropout(.5),
                                nn.ReLU(),
                                nn.Conv1d(32, 1, 3),
                                nn.Dropout(.5),
                                nn.ReLU(),
                                )
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, s):
        a, b, _, d, e = self._preproc(s)
        a, d = [self.batchNormMatrix(x) for x in [a, d]]
        b, e =[(x - x.min()) / (x.max() - x.min()) for x in [b, e]]
        X, Y = [torch.cat((x, y.unsqueeze(1)), 1) for x, y in zip([a, d], [b, e])]
        X, Y = [x.unsqueeze(1) for x in [X, Y]]
        H, G = [self.core(x) for x in [X, Y]]
        H, G = [x.squeeze(1) for x in [H, G]]
        S = H @ G.T
        action_scores = S.mean(0)
        return F.softmax(action_scores, dim=-1)
    
    
    def _preproc(self, s):
        return [torch.FloatTensor(item) for item in s]