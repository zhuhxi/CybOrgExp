import torch
import numpy as np
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from CybORG.Agents.Wrappers import *
import sys
sys.path.append("/home/zhx/word/cage-challenge-1/")
from DQN.DQNAgent import DQNAgent, RNNDQNAgent
from os.path import expanduser, join, exists
from os import mkdir, listdir
import pandas as pd
PATH = '/home/zhx/word/cage-challenge-1/CybORG/CybORG/Shared/Scenarios/Scenario1b.yaml'
load_checkpoint = None

np.random.seed(0)
torch.manual_seed(0)

DIR = "mutli_agent_rnnddqn_models"
if not exists(DIR):
    mkdir(DIR)

# alternate between both agents (ensure you are using the right DIR however)
CYBORG = CybORG(PATH, 'sim', agents={
    'Red': RedMeanderAgent,
    # 'Red': B_lineAgent,
    # 'Red': SleepAgent,
})

# check if cuda is available
def cuda():
    print("CUDA: " + str(torch.cuda.is_available()))



def train(agent_name="Blue", gamma=0.99, epsilon=1, lr=0.0001,
                 mem_size=1000, batch_size=64, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, n_episodes=100, len_episodes=32, lookback_steps=7, suffix=0, hid_size=64):


    agent_counter = 0
    agent_list = [
        {'Red': RedMeanderAgent},
        {'Red': B_lineAgent},
        {'Red': SleepAgent},
    ]

    env = ChallengeWrapper(agent_name="Blue",
                           env=CybORG(PATH, 'sim', agents=agent_list[agent_counter]))

    agent = RNNDQNAgent(gamma=gamma, epsilon=epsilon, lr=lr, lookback_steps=lookback_steps,
                     input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=mem_size, eps_min=eps_min,
                     batch_size=batch_size, replace=replace, eps_dec=eps_dec,
                     chkpt_dir=DIR, algo=f'RNNDDQNAgent_{suffix}',
                     env_name='Scenario1b', hid_size=hid_size)

    best_score = -np.inf

    n_steps = 0
    scores, eps_history, steps_array = [], [], []
    action_space = env.action_space

    try:
        for i in range(n_episodes):
            score = 0

            observation = env.reset()
            observation_buffer = np.zeros((agent.lookback_steps, len(observation)))
            observation_buffer[-1] = observation
            print(agent_list[agent_counter % len(agent_list)])

            for j in range(len_episodes):
                action = agent.get_action(observation_buffer, action_space=action_space)
                observation_, reward, done, info = env.step(action=action)
                score += reward
                if not load_checkpoint:
                    agent.store_transition(observation_buffer, action,
                                         reward, observation_, int(done))
                    agent.train()

                observation = observation_

                # update_buffer
                observation_buffer[:-1] = observation_buffer[1:]
                observation_buffer[-1] = observation

                n_steps += 1
            scores.append(score)
            steps_array.append(n_steps)

            avg_score = np.mean(scores[-100:])
            print('episode: ', i,'score: ', score,
                 ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
                'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

            # keep track of best score to see if we are converging
            if avg_score > best_score:
                best_score = avg_score

            eps_history.append(agent.epsilon)

            # get new agent env
            agent_counter += 1
            env = ChallengeWrapper(agent_name="Blue",
                                   env=CybORG(PATH, 'sim', agents=agent_list[agent_counter % len(agent_list)]))
    except KeyboardInterrupt:
        agent.save_models()


    agent.save_models()



options = {
        "gamma": [0.99, 0.95, 0.90, 0.80, 0.75, 0.5],
        "epsilon": [1, 0.75, 0.5],
        "lr": [0.0001, 0.001, 0.01],
        "mem_size": [500, 1000, 10000],
        "eps_min": [0.1, 0.05, 0.01, 0.001],
        "batch_size": [32, 64, 128, 256],
        "replace": [200, 500, 1000, 5000],
        "len_episodes": [100, 30, 50],
        "eps_dec": [0.000005, 0.0005, 0.05],
        "n_episodes": [100, 1000, 1500],
        "lookback_steps": [16, 32],
        "hid_size": [64, 256, 1024]
}

def random_search(n_configs=100):
    done = len(listdir(DIR)) // 2
    for i in range(done, n_configs + done):
        print(f"\n\n\n\n\n {i} \n\n\n\n")
        params = {k: np.random.choice(v) for k, v in options.items()}
        params["agent_name"] = "Blue"
        params["suffix"] = i
        print(params)
        train(**params)
        result = pd.DataFrame(params, index=pd.DatetimeIndex([0]))
        result.to_csv("random_search_more.csv", mode="a", index=False, header=(i == 0))


# uncomment and comment the methods you want to run in main()
# if you want to train yourself make sure to set DIR to something new (and make a directory)
if __name__ == '__main__':
    random_search(n_configs=100)