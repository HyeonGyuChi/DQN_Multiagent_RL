import random
import operator
import numpy as np
import sys
import os

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import itertools
from tqdm import tqdm



class choleclandmarks:
    LEFT = 0
    RIGHT = 1
    A = [LEFT, RIGHT]
    A_DIFF = [-1, 1]
    
    def __init__(self, args):
        self.reward_mode = args['reward_mode']
        self.num_agents = args['agents_number']
        self.num_landmarks = 2
        
        self.grid_size = args['grid_size']
        self.agents_positions = []
        self.landmarks_positions = []

        self.agents_init_positions = []
        self.features = []
        # self.positions_idx = []

        self.num_episodes = 0
        self.terminal = False
    
        self.search_window_size = args['search_window_size']
        self.state_size = self.search_window_size * 2 # 2L
        
        self.loop_cnt = 0 # 등록된 모든 videolandmarks를 순회할때마다 +1
        self.landmark_idx = -1
        self.videofeatures = {}
        self.videolandmarks = {}
        '''
        0:
            vid: 1
            features: numpy
            start_idx: int
            end_idx: int
        1:
            vid: 1
            features: numpy
            start_idx: int
            end_idx: int
        2:
            vid: 2
            ...
        ...
        '''
    
        self._load_videolandmarks(label=3)
        
        self.positions_ridx = self._get_positions_ridx(method='FIS') # reference idx
        
    def __len__(self):
        return len(self.videolandmarks)
    
    def _load_videolandmarks(self, label=3):                
        
        # 비디오에서 label에 대한 시작과 끝 index 모두탐색
        def find_start_end_indices_all(input_array, target_value):
            # runlength
            # input_array = [1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 3, 3]
            # Runlength 인코딩 결과: [(1, 3), (2, 2), (1, 3), (2, 2), (3, 2)]
            
            ind = 0
            start_indices, end_indices = [], []
            for value, cnt in [(key, len(list(group))) for key, group in itertools.groupby(input_array)]:
                if value == target_value:
                    start_indices.append(ind)
                    end_indices.append(ind + cnt - 1)
                
                ind += cnt
            
            return start_indices, end_indices
        
        data_dir = '/raid/dataset/public/medical/Cholec80/features'
        cholec80 = Cholec80(data_dir)
        train_dataset = cholec80.data['train']
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

        l_idx = 0
        for batch in train_dataloader:
            features, y, vid = batch
            features, y, vid = features.squeeze(), y.squeeze(), vid.squeeze()
            features, y, vid = features.numpy(), y.numpy(), int(vid.numpy())
            
            print('vid:', vid ,'|', 'features shape:', features.shape) # [1, 1734, 7]
            # print(y.shape) # [1, 1734]
            # print(y)
            
            
            start_indices, end_indices = find_start_end_indices_all(y, target_value=label)
            print('start_indices:', start_indices, 'end_indices:', end_indices)
            # print(y[start_indices[0]: end_indices[0]+1])
            
            
            # video features
            self.videofeatures[vid] = features
            
            # video landmarks
            for i in range(len(start_indices)):
                if l_idx not in self.videolandmarks:
                    self.videolandmarks[l_idx] = {}
            
                self.videolandmarks[l_idx]['vid'] = vid
                self.videolandmarks[l_idx]['start_idx'] = start_indices[i]
                self.videolandmarks[l_idx]['end_idx'] = end_indices[i]
                l_idx += 1
        
    def _get_positions_ridx(self, method='FIS'):
        position_ridx = np.zeros(2)
        
        # 0~1 uniform # agents 상대위치 초기화
        if method == 'random':
            position_ridx = np.sort(np.random.rand(2))
            
        
        elif method == 'FIS':
            
            start_rids, end_rids = [], []
            for i in range(len(self.videolandmarks)):
                vid = self.videolandmarks[i]['vid']
                s_idx = self.videolandmarks[i]['start_idx']
                e_idx = self.videolandmarks[i]['end_idx']
                vlen = self.videofeatures[vid].shape[0]
                
                # relative ids
                start_rids.append(s_idx / (vlen - 1))
                end_rids.append(e_idx / (vlen - 1))
            
            s_sigma, s_mu = np.mean(start_rids), np.std(start_rids)
            e_sigma, e_mu = np.mean(end_rids), np.std(end_rids)
            
            # sample distribution
            position_ridx[0] = np.clip(s_sigma * np.random.randn(1) + s_mu, 0, 1)
            position_ridx[1] = np.clip(e_sigma * np.random.randn(1) + e_mu, 0, 1)
            position_ridx = np.sort(position_ridx)
            
            # print(s_sigma, s_mu)
            # print(e_sigma, e_mu)
            # print(position_ridx)

        return position_ridx
    
    def reset(self):  # initialize the world  
        self.landmark_idx += 1  

        # init landmark_idx
        # 등록된 모든 landmark를 순회하였을 경우
        if self.landmark_idx >= len(self.videolandmarks):            
            self.positions_ridx = self._get_positions_ridx(method='FIS')
            self.landmark_idx = 0
            self.loop_cnt += 1
            
            print('initialize the agents reference positions | {}'.format(self.positions_ridx))
            

        self.terminal = False
        [self.features, self.agents_positions, self.landmarks_positions] = self.set_positions_idx(self.landmark_idx)
        self.agents_init_positions = np.copy(self.agents_positions)
        # separate the generated position indices for walls, pursuers, and evaders
        # self.landmarks_positions_idx = self.positions_idx[0:self.num_landmarks]
        # self.agents_positions_idx = self.positions_idx[self.num_landmarks:self.num_landmarks + self.num_agents]

        # self.agents_positions_idx = self.positions_idx

        # map generated position indices to positions
        # self.landmarks_positions = [self.cells[pos] for pos in landmarks_positions_idx]
        # self.agents_positions = [self.cells[pos] for pos in agents_positions_idx]
        
        
        # Zero padding 추가
        # 13 // 2 = 6
        # 10 - 6 / 10 + 7 / 4, 17 / 4 5 6 7 8 9 10 11 12 13 14 15,16
        
        '''
        pad_size = self.search_window_size // 2
        
        # search_window_size = 21일때, 21 length의 클립중 앞에서 11번째있는게 center (=target idx) 로 사용
        if self.search_window_size % 2 == 1:
            padded_features = np.vstack([np.zeros((pad_size, self.features.shape[1])), self.features, np.zeros((pad_size + 1, self.features.shape[1]))])            
        else:
            padded_features = np.vstack([np.zeros((pad_size, self.features.shape[1])), self.features, np.zeros((pad_size, self.features.shape[1]))])
        
        print('features: ', padded_features.shape)
        print('agents_positions: ', self.agents_positions, type(self.agents_positions))
        print('landmarks_positions: ', self.landmarks_positions, type(self.landmarks_positions))
        
        start_state = padded_features[self.agents_positions[0]:self.agents_positions[0] + self.search_window_size, :]
        end_state = padded_features[self.agents_positions[1]:self.agents_positions[1] + self.search_window_size, :]
        initial_state = np.vstack([start_state, end_state])

        '''

        # initial_state = list(sum(self.landmarks_positions + self.agents_positions, ())) # just flatten
        
        initial_state = self.get_state()
        # print('initial_state: ', initial_state.shape)
    
        return initial_state
        
        
    def set_positions_idx(self, idx):
        videolandmarks = self.videolandmarks[idx]
        vid, start_idx, end_idx = videolandmarks['vid'], videolandmarks['start_idx'], videolandmarks['end_idx']
        features = self.videofeatures[vid]
        vlen = features.shape[0]
        
        # self.cells = features
        
        # cells = [(i, j) for i in range(0, self.grid_size) for j in range(0, self.grid_size)] 
        

        # agents position | random
        positions_idx = self.positions_ridx * vlen
        positions_idx = positions_idx.astype(int)
        positions_idx = np.clip(positions_idx, 0, vlen-1) # [0, vlen-1]

        landmark_idx = np.array([start_idx, end_idx])
        
        return [features, positions_idx, landmark_idx]
    
    def get_state(self):
        pad_size = self.search_window_size // 2
    
        # search_window_size = 21일때, 21 length의 클립중 앞에서 11번째있는게 center (=target idx) 로 사용
        if self.search_window_size % 2 == 1:
            padded_features = np.vstack([np.zeros((pad_size, self.features.shape[1])), self.features, np.zeros((pad_size + 1, self.features.shape[1]))])            
        else:
            padded_features = np.vstack([np.zeros((pad_size, self.features.shape[1])), self.features, np.zeros((pad_size, self.features.shape[1]))])
        
        # print('features: ', self.features.shape)
        # print('padded_features: ', padded_features.shape)
        # print('agents_positions: ', self.agents_positions, type(self.agents_positions))
        # print('landmarks_positions: ', self.landmarks_positions, type(self.landmarks_positions))
        
        start_state = padded_features[self.agents_positions[0]:self.agents_positions[0] + self.search_window_size, :]
        end_state = padded_features[self.agents_positions[1]:self.agents_positions[1] + self.search_window_size, :]
        state = np.vstack([start_state, end_state])
        
        return state
    
    
    def step(self, agents_actions):
        # update the position of agents
        # agents_actions = [0, 1] // list
        # agents_positions = [100, 200] // numpy
        
        # update agent, landmark position
        self.agents_positions = self.update_positions(self.agents_positions, agents_actions)
        
        # reward
        # inference: 모든 agents들이 GT에 도달했을 경우? Inference 에서는 r, l 로 왔다갔다 할 경우 종료?
        # train: agents들이 GT에 도달할 경우 a+1, 아닐경우 -1
        a = 1
        is_goal = self.landmarks_positions == self.agents_positions
        reward = np.where(is_goal, a+1, -1)
        
        # terminal case
        # 두 agents 모두 goal이면 terminate
        if np.all(is_goal):
            self.terminal = True
        else:
            self.terminal = False
            
        # new state
        new_state = self.get_state()
        
        # print('new_state:', new_state.shape)
        # print('is_goal:', is_goal)
        # print('reward:', reward)
        # print('self.terminal:', self.terminal)

        # [numpy, numpy, True or False]
        # [new_state, [2, -1], True or False]
        return [new_state, reward, self.terminal]
    
    
    def update_positions(self, pos_np, act_list:list):
        start_action, end_action = act_list[0], act_list[1]
        pos_act_applied = pos_np + np.array([self.A_DIFF[start_action], self.A_DIFF[end_action]])
        
        vlen = self.features.shape[0] # check over seq length
        pos_act_applied = np.clip(pos_act_applied, 0, vlen-1) # [0, vlen-1]
        
        # temporal constraint
        if pos_act_applied[0] >= pos_act_applied[1]:
            pos_act_applied[0] = pos_act_applied[1]
        
        final_positions = pos_act_applied
        
        return final_positions
    
    
    def action_space(self):
        return len(self.A)
    
    def feature_dim(self):
        return 2048
    
    
class Cholec80Helper(Dataset):
    def __init__(self, data_root, factor_sampling, data_p, dataset_split=None):
        assert dataset_split != None
        self.data_p = data_p
        assert data_root != ""
        self.data_root = os.path.abspath(data_root)
        self.number_vids = len(self.data_p)
        self.dataset_split = dataset_split
        self.factor_sampling = factor_sampling

    def __len__(self):
        return self.number_vids

    def __getitem__(self, index):
        p = os.path.join(self.data_root, self.data_p[index])
        vid = int(os.path.basename(p).split('_')[1]) # video_01_1.0fps.pkl -> 1
        unpickled_x = pd.read_pickle(p)
        stem = np.asarray(unpickled_x[0],
                          dtype=np.float32)[::self.factor_sampling]
        
        # average features (default 16)
        stem = self._average_feature(stem, clip_size=16)
        
        # y_hat = np.asarray(unpickled_x[1],
        #                    dtype=np.float32)[::self.factor_sampling]
        y = np.asarray(unpickled_x[2])[::self.factor_sampling]
        return stem, y, vid
    
    def _average_feature(self, stem, clip_size):
        # 입력 배열의 크기 확인
        assert stem.ndim == 2, "입력 배열은 2D여야 합니다."

        # clip_size가 0보다 커야 합니다.
        assert clip_size > 0, "clip_size는 0보다 커야 합니다."

        # 결과를 저장할 배열 초기화
        result_array = np.zeros_like(stem)

        # Zero padding 추가
        pad_size = clip_size // 2
        if clip_size % 2 == 1: # odd
            padded_input = np.vstack([np.zeros((pad_size, stem.shape[1])), stem, np.zeros((pad_size, stem.shape[1]))])
        else: # even
            padded_input = np.vstack([np.zeros((pad_size, stem.shape[1])), stem, np.zeros((pad_size + 1, stem.shape[1]))])

        # 입력 sequence를 1씩 옮겨가면서 clip_size만큼 feature를 average
        for i in range(stem.shape[0]):
            start_index = i
            end_index = i + clip_size
            selected_features = padded_input[start_index:end_index, :]
            averaged_features = np.mean(selected_features, axis=0)
            result_array[i, :] = averaged_features

        return result_array

        '''
        # 예제 사용
        stem = np.random.randn(2957, 2048)
        clip_size = 16
        result = average_features_with_padding(stem, clip_size)
        print("Averaged Features with Padding Shape:", result.shape)
        '''
    
class Cholec80():
    def __init__(self, data_dir, features_per_seconds=1, features_subsampling=25):
        self.class_labels = [
            "Preparation",
            "CalotTriangleDissection",
            "ClippingCutting",
            "GallbladderDissection",
            "GallbladderPackaging",
            "CleaningCoagulation",
            "GallbladderRetraction",
        ]
        # self.out_features = self.hparams.out_features
        self.features_per_seconds = features_per_seconds
        self.features_subsampling = features_subsampling
        factor_sampling = (int(25 / self.features_subsampling))
        print(
            f"Subsampling features: 25features_ps --> {self.features_subsampling}features_ps (factor: {factor_sampling})"
        )

        # RL논문에서는 40, 20, 20
        self.data_p = {}
        self.data_p["train"] = [(
            f"video_{i:02d}_{self.features_per_seconds:.1f}fps.pkl"
        ) for i in range(1, 41)] # range(1, 41)
        self.data_p["val"] = [(
            f"video_{i:02d}_{self.features_per_seconds:.1f}fps.pkl"
        ) for i in range(41, 49)]
        self.data_p["test"] = [(
            f"video_{i:02d}_{self.features_per_seconds:.1f}fps.pkl"
        ) for i in range(49, 81)]

        self.data = {}
        self.data_dir = data_dir
        for split in ["train", "val", "test"]:
            self.data[split] = Cholec80Helper(self.data_dir, 
                                              factor_sampling,
                                              self.data_p[split],
                                              dataset_split=split)

        print(
            f"train size: {len(self.data['train'])} - val size: {len(self.data['val'])} - test size:"
            f" {len(self.data['test'])}")



if __name__ == "__main__":
    
    # data_dir = '/raid/dataset/public/medical/Cholec80/features'
    # cholec80 = Cholec80(data_dir)
    # train_dataset = cholec80.data['train']
    # train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    
    
    # for batch in train_dataloader:
    #     features, y = batch
    #     features, y = features.squeeze(), y.squeeze()
        
    #     print(features.size()) # [1, 1734, 7]
    #     print(y.size()) # [1, 1734]
        
    # features, y = next(iter(train_dataloader)) # [1, 1734, 7] / [1, 1734, 2048] / [1, 1734]
    
    args = {
        'reward_mode': 0,
        'agents_number': 0,
        'grid_size': 0,
    }
    env = choleclandmarks(args)
    env._load_videolandmarks(label=5)
    
    env.reset()
    print(env.loop_cnt)
    print(env.landmark_idx)
    print(env.positions_ridx)
        
    for i in tqdm(range(3)): # vid
        for j in range(100): # step
            print('----{}----'.format(j))
            
            act_lst = [random.randrange(2), random.randrange(2)]
            print('act_lst:', act_lst)
            env.step(act_lst)
            print('==========\n')