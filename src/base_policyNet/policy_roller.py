import sys
sys.path.append("../../")
from gymenv_v2 import make_multiple_env
import torch
import numpy as np
from torch.distributions import Categorical
from torch.optim.lr_scheduler import StepLR
class Observer(object):
    def __init__(self, env_config):
        self.env = make_multiple_env(**env_config)
    def run_episode(self, agent):
        state, ep_reward, d = self.env.reset(), 0, False
        while not d:
            action = agent.select_action(state)
            state, reward, d, _ = self.env.step(action)
            agent.report_reward(reward, d)

class Agent(object):
    def __init__(self, training_config,observer, model):
        learning_rate = training_config['lr']
        gamma = training_config['gamma']
        self.entropy_coef = training_config['entropy_coef']
        self.observer = observer
        self.rewards = []
        self.gamma = gamma
        self.policy = model
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=training_config['step_size'], gamma=training_config['decay'])
        self.eps = np.finfo(np.float32).eps.item()
        self.save_log_probs = []
        self.save_probs = []
        
    def select_action(self, state):
        probs = self.policy(state)
        probability = probs + self.eps
        probability /= probability.sum()
        m = Categorical(probs)
        action = m.sample()
        self.save_log_probs.append(m.log_prob(action))
        self.save_probs.append(m.probs[action])
        return action.item()
    def report_reward(self, reward, d):
        if not d:
            self.rewards.append(reward)
        else:
            self.rewards.append(reward)
            self.rewards.append(np.NaN)
        
    def run_episode(self):
        self.observer.run_episode(self)
        
    def finish_episode(self):
        R, log_probs, probs  = 0, self.save_log_probs.copy(), self.save_probs.copy()
        
        rewards = []
        rewards_seqs = []
        rewards_seq = []
        for reward in self.rewards:
            if not np.isnan(reward):
                rewards.append(reward)
                rewards_seq.append(reward)
            else:
                rewards_seqs.append(rewards_seq)
                rewards_seq = []
        reward = min([sum(rewards_seq) for rewards_seq in rewards_seqs])
        
        self.rewards = []
        self.save_log_probs = []
        self.save_probs = []
        
        policy_loss, returns = [], []
        
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        
        for log_prob, prob, R in zip(log_probs, probs, returns):
            policy_loss.append(-log_prob * R - self.entropy_coef * (log_prob * prob))
        self.optimizer.zero_grad()
        
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return reward