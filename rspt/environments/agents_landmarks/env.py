import random
import operator
import numpy as np
import sys
import os

class agentslandmarks:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4
    A = [UP, DOWN, LEFT, RIGHT, STAY]
    A_DIFF = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

    def __init__(self, args, current_path):
        self.game_mode = args['game_mode']
        self.reward_mode = args['reward_mode']
        self.num_agents = args['agents_number']
        self.num_landmarks = self.num_agents # landmarks
        self.grid_size = args['grid_size']
        self.state_size = (self.num_agents + self.num_landmarks) * 2
        self.agents_positions = []
        self.landmarks_positions = []

        self.cells = []
        self.positions_idx = []

        self.num_episodes = 0
        self.terminal = False

    def set_positions_idx(self):

        cells = [(i, j) for i in range(0, self.grid_size) for j in range(0, self.grid_size)]

        positions_idx = []

        if self.game_mode == 0:
            # first enter the positions for the landmarks and then for the agents. If the grid is n*n, then the
            # positions are
            #  0                1             2     ...     n-1
            #  n              n+1           n+2     ...    2n-1
            # 2n             2n+1          2n+2     ...    3n-1
            #  .                .             .       .       .
            #  .                .             .       .       .
            #  .                .             .       .       .
            # (n-1)*n   (n-1)*n+1     (n-1)*n+2     ...   n*n+1
            # , e.g.,
            # positions_idx = [0, 6, 23, 24] where 0 and 6 are the positions of landmarks and 23 and 24 are positions
            # of agents
            positions_idx = []

        if self.game_mode == 1:
            positions_idx = np.random.choice(len(cells), size=self.num_landmarks + self.num_agents,
                                             replace=False)

        return [cells, positions_idx]

    def reset(self):  # initialize the world

        self.terminal = False
        [self.cells, self.positions_idx] = self.set_positions_idx()

        # separate the generated position indices for walls, pursuers, and evaders
        landmarks_positions_idx = self.positions_idx[0:self.num_landmarks]
        agents_positions_idx = self.positions_idx[self.num_landmarks:self.num_landmarks + self.num_agents]

        # map generated position indices to positions
        self.landmarks_positions = [self.cells[pos] for pos in landmarks_positions_idx]
        self.agents_positions = [self.cells[pos] for pos in agents_positions_idx]

        initial_state = list(sum(self.landmarks_positions + self.agents_positions, ())) # just flatten

        return initial_state

    def step(self, agents_actions):
        # update the position of agents
        self.agents_positions = self.update_positions(self.agents_positions, agents_actions)

        if self.reward_mode == 0:

            binary_cover_list = []

            for landmark in self.landmarks_positions:
                distances = [np.linalg.norm(np.array(landmark) - np.array(agent_pos), 1)
                             for agent_pos in self.agents_positions]

                min_dist = min(distances)

                if min_dist == 0:
                    binary_cover_list.append(min_dist)
                else:
                    binary_cover_list.append(1)

            # check the terminal case
            if sum(binary_cover_list) == 0:
                reward = 0
                self.terminal = True
            else:
                reward = -1
                self.terminal = False

        if self.reward_mode == 1:

            binary_cover_list = []

            for landmark in self.landmarks_positions:
                # 모든 agents들이 GT에 도달했을 경우? Inference 에서는 r, l 로 왔다갔다 할 경우 종료?
                distances = [np.linalg.norm(np.array(landmark) - np.array(agent_pos), 1)
                             for agent_pos in self.agents_positions]

                min_dist = min(distances)

                if min_dist == 0:
                    binary_cover_list.append(0)
                else:
                    binary_cover_list.append(1)

            reward = -1 * sum(binary_cover_list)
            # check the terminal case
            if reward == 0:
                self.terminal = True
            else:
                self.terminal = False

        if self.reward_mode == 2:

            # calculate the sum of minimum distances of agents to landmarks
            reward = 0
            for landmark in self.landmarks_positions:
                distances = [np.linalg.norm(np.array(landmark) - np.array(agent_pos), 1)
                             for agent_pos in self.agents_positions]

                reward -= min(distances)

            # check the terminal case
            if reward == 0:
                self.terminal = True

        new_state = list(sum(self.landmarks_positions + self.agents_positions, ()))

        return [new_state, reward, self.terminal]

    def update_positions(self, pos_list, act_list):
        positions_action_applied = []
        for idx in range(len(pos_list)):
            if act_list[idx] != 4:
                pos_act_applied = list(map(operator.add, pos_list[idx], self.A_DIFF[act_list[idx]]))
                # checks to make sure the new pos in inside the grid
                for i in range(0, 2):
                    if pos_act_applied[i] < 0:
                        pos_act_applied[i] = 0
                    if pos_act_applied[i] >= self.grid_size:
                        pos_act_applied[i] = self.grid_size - 1
                positions_action_applied.append(tuple(pos_act_applied))
            else:
                positions_action_applied.append(pos_list[idx])

        final_positions = []

        for pos_idx in range(len(pos_list)):
            if positions_action_applied[pos_idx] == pos_list[pos_idx]:
                final_positions.append(pos_list[pos_idx])
            elif positions_action_applied[pos_idx] not in pos_list and positions_action_applied[
                pos_idx] not in positions_action_applied[
                                0:pos_idx] + positions_action_applied[
                                             pos_idx + 1:]:
                final_positions.append(positions_action_applied[pos_idx])
            else:
                final_positions.append(pos_list[pos_idx])

        return final_positions

    def action_space(self):
        return len(self.A)