import sys
sys.path.append("..")
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
        hidden = None
        while not d:
            # send the state to the agent to get an action
            action, hidden = agent.select_action(state, hidden)

            # apply the action to the environment, and get the reward
            state, reward, d, _ = self.env.step(action)
            # report the reward to the agent for training purpose
            agent.policy.epsilon *= agent.policy.epsilon_decay
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
#         self.scheduler = StepLR(self.optimizer, step_size=4, gamma=0.1)
        self.eps = np.finfo(np.float32).eps.item()
        self.save_log_probs = []
        self.save_probs = []
        
    def select_action(self, state, hidden):
        probs, hidden = self.policy(state, hidden)
        try:
            m = Categorical(probs)
        except ValueError:
            probs = torch.rand(probs.shape)
            probs /= probs.sum()
            m = Categorical(probs)
        action = m.sample()
        self.save_log_probs.append(m.log_prob(action))
        self.save_probs.append(m.probs[action])
        return action.item(), hidden
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
            # policy_loss.append(-log_prob * R)
            policy_loss.append(-log_prob * R - self.entropy_coef * (log_prob * prob))
        self.optimizer.zero_grad()
        
        policy_loss = torch.stack(policy_loss).sum()
#         print(policy_loss)
        policy_loss.backward()
        self.optimizer.step()
#         self.scheduler.step()
        return reward