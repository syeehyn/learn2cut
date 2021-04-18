from src.attention_free.cnn_policy import Policy
from src.attention_free.policy_roller import Observer, Agent
import json
import sys
import wandb
wandb.login()
if __name__ == "__main__":
    targets = sys.argv[1:]
    if 'easy' in targets:
        env_config = {
        "load_dir"        : 'instances/train_10_n60_m60',
        "idx_list"        : list(range(10)),
        "timelimit"       : 50,
        "reward_type"     : 'obj'
        }
        env_name = 'training-easy'

    elif 'hard' in targets:
        env_config = {
            "load_dir"        : 'instances/train_100_n60_m60',
            "idx_list"        : list(range(99)),
            "timelimit"       : 50,
            "reward_type"     : 'obj'
        }
        env_name = 'training-hard'
    else:
        sys.exit()
    training_config = {
            'lr': 1e-3,
            'gamma': .95,
            'num_revisit': 1,
        }
    for entropy_coef in [1, 1e-1, 1e-2]:
        for step in [2, 4, 8, 12]:
            training_config['entropy_coef'] = entropy_coef
            training_config['step_size'] = step
            run=wandb.init(project="finalproject", entity="ieor-4575", tags=[env_name], reinit=True)
            model = Policy(60)

            observer = Observer(env_config)
            agent = Agent(training_config, observer, model)

            for iteration in range(40):
                for _ in range(training_config['num_revisit']):
                    agent.run_episode()
                reward = agent.finish_episode()
                print(f'iter: {iteration}, training reward: {reward}')
                wandb.log({"Training reward" : reward,
                            "entropy_coef": entropy_coef,
                            'step': step
                })