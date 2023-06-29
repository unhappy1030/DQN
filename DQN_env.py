import pandas as pd
import numpy as np
import datetime
class Environment:
    def __init__(self, state_file_path):
        self.states = pd.read_csv(state_file_path).values
        self.states[:, 1] = self.states[:, 1].astype('float32')
        self.states[:, 1] = 0
        self.states[:, 0] = self.states[:, 0].astype('datetime64[ms]')
        self.states[:, 0] = [datetime.datetime.timestamp(dt) for dt in self.states[:, 0]]
        self.states[:, 0] = self.states[:, 0].astype('float32')
        self.states[:, 1] = 0
        self.states = np.array(self.states)
        self.states = self.states.astype('float32')
        self.current_step = 0
        self.actions = np.array([0,1,2])
        self.a1_st = -1
        self.a2_st = -1
    def _read_file(self, file_path):
        with open(file_path, 'r') as f:
            return list(map(float, f.readlines()))

    def get_state(self):
        return self.states[self.current_step]
    
    def get_action(self):
        return self.actions
    
    def get_reward(self):
        if self.a1_st == -1:
            return 0.5
        else:
            original_value = (self.states[self.current_step][1] - self.states[self.a1_st][1])
            scaled_value = self.scale_reward(original_value)
            return scaled_value
    def step(self, action = 0):
        if self.current_step >= len(self.states):
            self.states[self.current_step], self.get_reward(), True
        if action == 1:
            self.a1_st = self.current_step
            self.next_step()
            return self.states[self.current_step], 0, False
        elif action == 0:
            self.next_step()
            return self.states[self.current_step], self.get_reward(), False
        else:
            self.next_step()
            return self.states[self.current_step], self.get_reward(), True
    def next_step(self):
        self.current_step += 1
    
    def reset(self):
        self.current_step = 0
    def scale_reward(self, reward):
        if reward <= -0.3:
            scaled_reward = 0.0
        elif reward <= 0:
            scaled_reward = (reward+0.3) / 3 * 5
        elif reward <= 0.01:
            scaled_reward = (reward * 50) + 0.5
        elif reward <= 0.012:
            scaled_reward = 1.0
        elif reward <= 0.3:
            scaled_reward = ((reward * -1) + 0.3) / 288 * 500 + 0.5
        else:
            scaled_reward = 0.5
        return scaled_reward
if __name__ == "__main__":
    filename = '006380'
    path = '/Users/white/Desktop/valak/Valak/Main_ValaK/DQN_project/data/' + filename + '_ST.csv'
    env = Environment(path)