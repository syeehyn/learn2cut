import sys
sys.path.append("../../")
from gymenv_v2 import make_multiple_env
import torch
import numpy as np
from torch.distributions import Categorical
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

class Observer(object):
    def __init__(self, env_config):
        self.env = make_multiple_env(**env_config)
    def run_episode(self, agent):
        state, d = self.env.reset(), False
        prev = None
        while not d:
            action, prev = agent.select_action(state, prev)
            state, reward, d, _ = self.env.step(action)
            agent.report_reward(reward)

class Agent(object):
    def __init__(self,observer, model, training_config):
        learning_rate = training_config['lr']
        self.gamma = training_config['gamma']
        self.sigma = training_config['sigma']
        self.model_id = training_config['model_id']
        self.num_revisit = training_config['num_revisit']
        self.observer = observer
        self.rewards = []
        self.policy = model
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, gamma=.3, step_size=5)
        self.eps = np.finfo(np.float32).eps.item()
        self.save_log_probs = []
        self.save_entropy = []
        self.best_reward = 0
        self.config = training_config
        
    def select_action(self, state, prev):
        probs, prev = self.policy(state, prev)
        
        m = Categorical(probs)
        action = m.sample()
        self.save_log_probs.append(m.log_prob(action))
        self.save_entropy.append(m.entropy().mean())
        return action.item(), prev
    def report_reward(self, reward):
        self.rewards.append(reward)
        
    def run_episode(self):
        self.observer.run_episode(self)
        
    def finish_episode(self):        
        R, log_probs, entropy  = 0, self.save_log_probs.copy(), self.save_entropy.copy()
        rewards = self.rewards
        reward = sum(rewards)
        self.rewards = []
        self.save_log_probs = []
        self.save_entropy = []
        policy_loss, returns = [], []
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        returns = returns + (self.sigma) * torch.randn(returns.shape)
        returns = F.relu(returns)
        entropy = torch.stack(entropy)
        log_probs = torch.stack(log_probs)
        policy_loss = - log_probs * returns + entropy
        self.optimizer.zero_grad()
        policy_loss = policy_loss.sum()
        policy_loss.backward()
        self.optimizer.step()
        return reward