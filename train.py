import argparse
import uuid
from src.seq2seq_policyNet.policy import Policy as seq2seq_policy
from src.seq2seq_policyNet.policy_roller import Observer as seq2seq_observer
from src.seq2seq_policyNet.policy_roller import Agent as seq2seq_agent
from src.attention_policyNet.policy import Policy as attention_policy
from src.attention_policyNet.policy_roller import Observer as attention_observer
from src.attention_policyNet.policy_roller import Agent as attention_agent

import wandb
import json
import os
import numpy as np
import torch

parser = argparse.ArgumentParser(description='learn2cut')
parser.add_argument('-m','--mode', help='hard, easy. hard mode or easy mode', required=True)
parser.add_argument('-n','--num_trajectories', help='int, number of trajectories', required=True)
parser.add_argument('-a', '--algo', help="attention, seq2seq type of algorithms", required=True)
parser.add_argument('-i', '--num_iters', help='number of iterations')

path_config = {
    'log_path': './log/log.json',
    'model_path': './trained_models'
}


def main(mode, num_visit, algo, num_iters):
    if mode == 'easy':
        env_config = {
        "load_dir"        : 'instances/train_10_n60_m60',
        "idx_list"        : list(range(10)),
        "timelimit"       : 50,
        "reward_type"     : 'obj'
        }
        env_name = 'training-easy'
    elif mode == 'hard':
        env_config = {
            "load_dir"        : 'instances/train_100_n60_m60',
            "idx_list"        : list(range(99)),
            "timelimit"       : 50,
            "reward_type"     : 'obj'
        }
        env_name = 'training-hard'
    else:
        return
    if algo == 'attention':
        training_config = {
                'lr': 1e-2,
                'gamma': .95,
                'num_revisit': num_visit,
                'sigma': 1e-3,
                'cnn_hidden': [8, 16],
                'rnn_hidden': 32,
                'dropout': .3,
                'model_id': str(uuid.uuid1()) + '.pt'
        }
        model = attention_policy(60, training_config['cnn_hidden'], 
                training_config['rnn_hidden'],
                training_config['dropout']
            )
        observer = attention_observer(env_config)
        agent = attention_agent(observer, model, training_config)
    elif algo == 'seq2seq':
        training_config = {
                'lr': 1e-2,
                'gamma': .95,
                'num_revisit': num_visit,
                'sigma': 1e-3,
                'arch': [32, 64, 32],
                'dropout': .3,
                'model_id': str(uuid.uuid1()) + '.pt'
            }
        model = seq2seq_policy(60, training_config['arch'], training_config['dropout'])
        observer = seq2seq_observer(env_config)
        agent = seq2seq_agent(observer, model, training_config)
    else:
        raise NotImplementedError
    
    run=wandb.init(project="finalproject", entity="ieor-4575", tags=[env_name], reinit=True)
    fixedWindow=10
    movingAverage=0
    rrecord = []
    for iteration in range(num_iters):
        model_weights = agent.policy.state_dict().copy()
        models = []
        rewards = []
        for _ in range(training_config['num_revisit']):
            agent.run_episode()
            reward = agent.finish_episode()
            rewards.append(reward)
            models.append(agent.policy.state_dict().copy())
            agent.policy.load_state_dict(model_weights)
        reward = np.mean(rewards)
        for key in model_weights:
            model_weights[key] = torch.mean(torch.stack([model[key] * 1.0 for model in models]), 0)
        rrecord.append(reward)
        if max(rrecord) == reward:
            torch.save(agent.policy.state_dict(), os.path.join(path_config['model_path'] ,agent.model_id))
            with open(path_config['log_path']) as f:
                log = json.load(f)
            agent.config['best_reward'] = reward
            log[agent.model_id] = agent.config
            with open(path_config['log_path'], 'w') as f:
                json.dump(log, f)
        agent.policy.load_state_dict(model_weights)
        movingAverage=np.mean(rrecord[len(rrecord)-fixedWindow:len(rrecord)-1])
        wandb.log({"Training reward" : reward, "movingAverage":movingAverage })
    run.summary['args'] = agent.config

if __name__ == '__main__':
    args = vars(parser.parse_args())
    mode = args['mode']
    num_visit = int(args['num_trajectories'])
    algo = args['algo']
    if args['num_iters'] != None:
        num_iters = int(args['num_iters'])
    else:
        num_iters = 20
    wandb.login()
    main(mode, num_visit, algo, num_iters)