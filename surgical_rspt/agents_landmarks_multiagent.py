import numpy as np
import os
import random
import argparse
import pandas as pd
from environments.agents_landmarks.env import choleclandmarks
from dqn_agent import Agent
import glob
from tqdm import tqdm
import torch

ARG_LIST = ['learning_rate', 'optimizer', 'memory_capacity', 'batch_size', 'target_frequency', 'maximum_exploration',
            'max_timestep', 'first_step_memory', 'replay_steps', 'number_nodes', 'target_type', 'memory',
            'prioritization_scale', 'agents_number', 'grid_size', 'game_mode', 'reward_mode', 'search_window_size']


def get_name_brain(args, idx):

    file_name_str = '_'.join([str(args[x]) for x in ARG_LIST])

    return './results_agents_landmarks/fis/phase4/weights_files/' + file_name_str + '_' + str(idx) + '.h5'


def get_name_rewards(args):

    file_name_str = '_'.join([str(args[x]) for x in ARG_LIST])

    return './results_agents_landmarks/fis/phase4/rewards_files/' + file_name_str + '.csv'


def get_name_timesteps(args):

    file_name_str = '_'.join([str(args[x]) for x in ARG_LIST])

    return './results_agents_landmarks/fis/phase4/timesteps_files/' + file_name_str + '.csv'

def get_name_positions(args):

    file_name_str = '_'.join([str(args[x]) for x in ARG_LIST])

    return './results_agents_landmarks/fis/phase4/positions_files/' + file_name_str + '.csv'


class Environment(object):

    def __init__(self, arguments):
        current_path = os.path.dirname(__file__)  # Where your .py file is located
        self.env = choleclandmarks(arguments)
        self.episodes_number = arguments['episode_number']
        self.max_ts = arguments['max_timestep']
        self.test = arguments['test']
        self.filling_steps = arguments['first_step_memory']
        self.steps_b_updates = arguments['replay_steps']
        self.max_random_moves = arguments['max_random_moves']

        self.num_agents = arguments['agents_number']
        self.game_mode = arguments['game_mode']
        self.grid_size = arguments['grid_size']

    def run(self, agents, file1, file2, file3):

        total_step = 0
        rewards_list = []
        timesteps_list = []
        infos_list = []
        positions_list = []
        max_score = float('-inf')
        for episode_num in range(self.episodes_number):
            
            '''
            state = self.env.reset()
            
            # print('state:', state, type(state)) # it agents:2 [3, 0, 3, 3, 2, 1, 1, 1] -> landmark: (3,0), (3,3) | agents: (2,1), (1,1)
            
            random_moves = random.randint(0, self.max_random_moves)

            # create randomness in initial state
            for _ in range(random_moves):
                actions = [4 for _ in range(len(agents))]
                state, _, _ = self.env.step(actions)
            
            # converting list of positions to an array
            # state = np.array(state)
            # state = state.ravel() 
            '''

            for vli in range(len(self.env)): # vli = video landmark index
                state = self.env.reset()
                
                done = False
                reward_all = 0
                time_step = 0
                while not done and time_step < self.max_ts:
                    actions = []
                    # for agent in tqdm(agents, desc='1'):
                    for agent in agents:
                        actions.append(agent.greedy_actor(state))
                    next_state, reward, done = self.env.step(actions) # agent 가 action을 한 다음 state
                    # converting list of positions to an array
                    # next_state = np.array(next_state)
                    # next_state = next_state.ravel()
                    
                    # print('next_state, reward, done')
                    # print(type(next_state))
                    # print(type(reward))
                    # print(type(done))
                    # print(next_state.shape, reward.shape, done)

                    if not self.test:
                        # for agent in tqdm(agents, desc='2'):
                        for agent in agents:
                            agent.observe((state, actions, reward, next_state, done)) # 위에서 greedy_Actor로 뽑은 agents결 행동을 memory에 저장
                            if total_step >= self.filling_steps: # default = 0
                                agent.decay_epsilon()
                                if time_step % self.steps_b_updates == 0: # update agents
                                    agent.replay()
                                agent.update_target_model()
                            
                        if time_step % 50 == 0:
                            print('episode_num:', episode_num, 'video_lanmark_index:', vli, 'time_step:', time_step, 'epsilon:', agent.epsilon)
                        

                    total_step += 1
                    time_step += 1
                    state = next_state
                    reward_all += reward

                rewards_list.append(reward_all)
                timesteps_list.append(time_step)
                infos_list.append((episode_num, vli, agent.epsilon))
                positions_list.append(np.hstack([self.env.features.shape[0], \
                        self.env.positions_ridx, self.env.agents_init_positions, \
                        self.env.agents_positions, self.env.landmarks_positions]))

                print("Episode {p}, Score: {s}, Final Step: {t}, Goal: {g}".format(p=episode_num, s=reward_all,
                                                                                t=time_step, g=done))
                # print('self.test', self.test)
                
                # print(self.env.positions_ridx)
                # print(self.env.agents_init_positions)
                # print(self.env.agents_positions)
                # print(self.env.landmarks_positions)
                # print(self.env.features.shape[0])
                
                if not self.test:
                    df = pd.DataFrame(np.hstack([np.array(infos_list), np.array(rewards_list)]), columns=['episode', 'vli', 'epsilon', 'score1', 'score2'])
                    df['reward_all'] = df['score1'] + df['score2']                        
                    
                    df.to_csv(file1)

                    df = pd.DataFrame(timesteps_list, columns=['steps'])
                    df.to_csv(file2)
                    
                    df = pd.DataFrame(np.array(positions_list), columns=['vlen', \
                                                                            'agents_init_ridx1', 'agents_init_ridx2', \
                                                                            'agents_init_idx1', 'agents_init_idx2', \
                                                                            'agents_final_idx1', 'agents_final_idx2', \
                                                                            'landmarks_idx1', 'landmarks_idx2'])

                    df.to_csv(file3)
                    
                    if total_step >= self.filling_steps:
                        if reward_all.sum() > max_score:
                            for agent in agents:
                                agent.brain.save_model()
                            max_score = reward_all.sum()


if __name__ =="__main__":

    parser = argparse.ArgumentParser()
    # DQN Parameters
    parser.add_argument('-e', '--episode-number', default=100000, type=int, help='Number of episodes') # 1000000
    parser.add_argument('-l', '--learning-rate', default=0.0003, type=float, help='Learning rate')
    parser.add_argument('-op', '--optimizer', choices=['Adam', 'RMSProp'], default='Adam',
                        help='Optimization method')
    parser.add_argument('-m', '--memory-capacity', default=10000, type=int, help='Memory capacity') # 1000000
    parser.add_argument('-b', '--batch-size', default=128, type=int, help='Batch size')
    parser.add_argument('-t', '--target-frequency', default=10000, type=int,
                        help='Number of steps between the updates of target network') # 10000
    parser.add_argument('-x', '--maximum-exploration', default=100000, type=int, help='Maximum exploration step') # 1000000 # agents가 decay_epsilon할때 설정값도달하기 전까지 MAX부터 계속내리다가 넘어가면 epsilon=MIN(0.1)로 변경 // 각 epsidoe 마다 초기화되는게 아니라 총 agnets의 step
    parser.add_argument('-fsm', '--first-step-memory', default=10000, type=float,
                        help='Number of initial steps for just filling the memory')
    parser.add_argument('-rs', '--replay-steps', default=4, type=float, help='Steps between updating the network')
    parser.add_argument('-nn', '--number-nodes', default=256, type=int, help='Number of nodes in each layer of NN')
    parser.add_argument('-tt', '--target-type', choices=['DQN', 'DDQN'], default='DQN')
    parser.add_argument('-mt', '--memory', choices=['UER', 'PER'], default='PER')
    parser.add_argument('-pl', '--prioritization-scale', default=0.5, type=float, help='Scale for prioritization')

    parser.add_argument('-gn', '--gpu-num', default='2', type=str, help='Number of GPU to use')
    parser.add_argument('-test', '--test', action='store_true', help='Enable the test phase if "store_false"')

    # Game Parameters
    parser.add_argument('-k', '--agents-number', default=2, type=int, help='The number of agents')
    parser.add_argument('-g', '--grid-size', default=4, type=int, help='Grid size')
    parser.add_argument('-ts', '--max-timestep', default=200, type=int, help='Maximum number of timesteps per episode') # 100
    parser.add_argument('-gm', '--game-mode', choices=[0, 1], type=int, default=1, help='Mode of the game, '
                                                                                        '0: landmarks and agents fixed, '
                                                                                        '1: landmarks and agents random ')

    parser.add_argument('-rw', '--reward-mode', choices=[0, 1, 2], type=int, default=1, help='Mode of the reward,'
                                                                                             '0: Only terminal rewards'
                                                                                             '1: Partial rewards '
                                                                                             '(number of unoccupied landmarks'
                                                                                             '2: Full rewards '
                                                                                             '(sum of dinstances of agents to landmarks)')

    parser.add_argument('-rm', '--max-random-moves', default=0, type=int,
                        help='Maximum number of random initial moves for the agents')
    
    parser.add_argument('-sws', '--search-window-size', default=21, type=int,
                        help='window search size for the agents')


    args = vars(parser.parse_args())
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_num']

    env = Environment(args)

    state_size = env.env.state_size # 42 (==sequience size, 2L)
    action_space = env.env.action_space() # 2
    feature_dim = env.env.feature_dim() # 2048

    all_agents = []
    for b_idx in range(args['agents_number']):

        brain_file = get_name_brain(args, b_idx)
        os.makedirs(os.path.dirname(brain_file), exist_ok=True)
        
        all_agents.append(Agent(state_size, action_space, feature_dim, b_idx, brain_file, args))

    rewards_file = get_name_rewards(args)
    timesteps_file = get_name_timesteps(args)
    positions_file = get_name_positions(args)
    
    os.makedirs(os.path.dirname(rewards_file), exist_ok=True)
    os.makedirs(os.path.dirname(timesteps_file), exist_ok=True)
    os.makedirs(os.path.dirname(positions_file), exist_ok=True)
    
    env.run(all_agents, rewards_file, timesteps_file, positions_file)