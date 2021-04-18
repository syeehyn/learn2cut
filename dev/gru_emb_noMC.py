import sys
sys.path.append('../')
from gymenv_v2 import make_multiple_env
import numpy as np


import torch
from torch import nn
import torch.nn.functional as F

# import wandb
# run=wandb.init(project="finalproject", entity="ieor-4575", tags=["training-easy"])

easy_config = {
    "load_dir"        : '../instances/train_10_n60_m60',
    "idx_list"        : list(range(10)),
    "timelimit"       : 50,
    "reward_type"     : 'obj'
}
env = make_multiple_env(**easy_config) 


class policyNet(nn.Module):
    def __init__(self, size_dim, hidden_size, output_size):
        super(policyNet, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(size_dim, size_dim)
        self.fc1 = nn.Linear(size_dim-1, size_dim)
        
        self.gru = nn.GRU(size_dim, hidden_size, batch_first=True)
        
        self.hidden_combine = nn.Linear(hidden_size*2, hidden_size)
        self.dropout = nn.Dropout(p = .2)

        self.out = nn.Linear(hidden_size, output_size)
        
        
#         self.fc1 = nn.Linear(size_dim, hidden)
    def forward(self, s, hidden):
        A, b, c, E, d = self._preproc(s)
        Ab = torch.hstack((A, b.unsqueeze(1)))
        Ed = torch.hstack((E, d.unsqueeze(1)))
        
        # c = self.embedding(c)
        
        # Ab = Ab @ c.T
        
        # Ab = F.relu(self.fc1(Ab))
        h, h_hidden = self.gru(Ab.unsqueeze(0), hidden.unsqueeze(0))
        
        g, g_hidden = self.gru(Ed.unsqueeze(0), hidden.unsqueeze(0))
        
        h = self.dropout(h)
        g = self.dropout(g)

        h = self.out(h.squeeze(0))
        g = self.out(g.squeeze(0))
        
        hidden = torch.cat((h_hidden.squeeze(0), g_hidden.squeeze(0)), 1)
        
        hidden = self.hidden_combine(F.relu(hidden))
        
        S = torch.mean(h @ g.T, 0)
        
        
        return F.log_softmax(S, dim=-1), hidden
        
        
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
        
    def _preproc(self, s):
        min1 = min(s[0].min(), s[-2].min())
        max1 = max(s[0].max(), s[-2].max())
        min2 = min(s[1].min(), s[-1].min())
        max2 = max(s[1].max(), s[-1].max())

        A = torch.FloatTensor((s[0] - min1) / (max1 - min1))
        E = torch.FloatTensor((s[-2] - min1) / (max1 - min1))
        b = torch.FloatTensor((s[1] - min2) / (max2 - min2))
        d = torch.FloatTensor((s[-1] - min2) / (max2 - min2))
        return [A, b, torch.LongTensor(s[2]), E, d]
def discounted_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_sum = 0
    for i in reversed(range(0,len(r))):
        discounted_r[i] = running_sum * gamma + r[i]
        running_sum = discounted_r[i]
    return torch.FloatTensor(discounted_r)

if __name__ == "__main__":
    env = make_multiple_env(**easy_config)
    N = 60
    alpha = 1e-2
    iterations = 20
    gamma = .8

    policy = policyNet(N+1, 128, 64)
    hidden = policy.initHidden()
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=alpha)
    policy_scheduler = torch.optim.lr_scheduler.ExponentialLR(policy_optimizer, gamma=0.5)
    rrecord = []
    for ite in range(iterations):
        obss = []
        acts = []
        rews = []
        s = env.reset()
        d = False
        t = 0
        repisode = 0
        while not d:

            with torch.no_grad():
                prob, _ = policy(s, hidden)
                prob = torch.exp(prob)
                prob /= prob.sum()
                
            a = np.random.choice(s[-1].size, p = prob.numpy(), size=1)

            obss.append(s)

            s, r, d, _ = env.step(list(a))

            # print('episode', ite, 'step', t, 'reward', r, 'action space size', s[-1].size, 'action', a[0])

            acts.append(a)
            rews.append(r)

            t += 1
            repisode += r

        rrecord.append(np.sum(rews))

        v_hat = discounted_rewards(rews, gamma)
        criterion = []

        # errs = torch.distributions.normal.Normal(0, 1)
        for obs, act, v in zip(obss, acts, v_hat):
            prob, hidden = policy(obs, hidden)
            prob_selected = prob[act]
            hidden = hidden.detach()
            loss = - (v) * prob_selected

            policy_optimizer.zero_grad()
            loss.backward()
            policy_optimizer.step()
            policy_scheduler.step()
            criterion.append(loss.item())
            
        print(f'loss: {np.mean(criterion)}')
        print(f'Training reward: {repisode}')
        # wandb.log({ "Training reward" : repisode})