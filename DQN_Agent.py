import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dropout, Flatten
import pandas as pd
import numpy as np
import random
from collections import deque
import os
from DQN_env import Environment
import matplotlib.pyplot as plt
import random
import traceback
import DQN_Wing
class DQNAgent:
    def __init__(self, state_file_path):
        # 환경을 초기화합니다.
        self.env = Environment(state_file_path)

        # 상태와 행동의 크기를 정의합니다.
        self.state_size = self.env.states.shape[1]
        self.action_size = len(self.env.actions)

        # DQN 하이퍼파라미터를 정의합니다.
        self.gamma = 0.95          # 감가율
        self.epsilon = 1.0         # 탐험 비율
        self.epsilon_min = 0.01    # 탐험 비율의 최솟값
        self.epsilon_decay = 0.995 # 탐험 비율 감쇠 상수
        self.learning_rate = 0.001 # 학습률
        
        self.n_episodes=10
        
        self.reward_list= []
        self.reward_sum = []
        self.moving_avg_list = []
        
        # replay memory를 정의합니다.
        self.memory = deque(maxlen=1000)
        model_path = "/Users/white/Desktop/valak/Valak/Main_ValaK/DQN_project/model_5"
        if not os.path.isfile(model_path):  # 파일이 존재하지 않으면
            self.q_network = self._build_network()
            self.target_network = self._build_network()
        else:  # 파일이 이미 존재하면
            self.q_network = tf.keras.models.load_model(model_path+'/q_network.h5')
            self.target_network = tf.keras.models.load_model(model_path+'/target_network.h5')
        self.update_target_network()
    def change_env(self, env_path):
        self.env = Environment(env_path)
    def _build_network(self):
        # Dueling 아키텍처를 갖는 Q 네트워크
        input_shape = (self.state_size,)
        inputs = Input(shape=input_shape)
        hidden = Dense(64, activation='relu')(inputs)
        
        # 상태 값 타워
        state_value = Dense(64, activation='relu')(hidden)
        state_value = Dense(1, activation='linear')(state_value)
        
        # 액션 장점 타워
        action_advantage = Dense(64, activation='relu')(hidden)
        action_advantage = Dense(self.action_size, activation='linear')(action_advantage)
        
        # 상태 값과 액션 장점을 결합하여 Q 값 획득
        q_values = Add()([state_value, action_advantage])
        
        model = Model(inputs=inputs, outputs=q_values)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        
        return model
    def train(self, batch_size=32):
        for episode in range(self.n_episodes):
            self.env.reset()
            state = self.env.states[self.env.current_step]
            done = False
            score = 0
            
            while not done:
                action = self.act(state)
                next_state, reward, done = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                score = reward
                if len(self.memory) > batch_size:
                    self.replay(batch_size)
            self.reward_list.append(score)
            epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
            
            print(f"Episode {episode + 1} Score: {score:.5f} Epsilon: {epsilon:.2f}")

    def update_target_network(self):
        # target network를 업데이트합니다.
        self.target_network.set_weights(self.q_network.get_weights())

    def remember(self, state, action, reward, next_state, done):
        # replay memory에 transition을 추가합니다.
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # epsilon-greedy 방식으로 액션 선택
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        # 리플레이 메모리에서 batch_size만큼의 전환을 랜덤하게 선택하여 학습합니다.
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = next_state.astype(float)
                next_state = np.expand_dims(next_state, axis=0)
                q_next_state = self.target_network.predict(next_state)
                max_q_next_state = np.max(q_next_state)
                target = reward + self.gamma * max_q_next_state.astype('float32')
            state = np.expand_dims(state, axis=0)
            target_f = self.q_network.predict(state)
            target_f[0][action] = target
            states.append(state)
            targets.append(target_f)
        states = np.reshape(states, (batch_size, self.state_size))
        targets = np.reshape(targets, (batch_size, self.action_size))
        self.q_network.fit(np.array(states), np.array(targets), epochs=5, verbose=0)


    def decay_epsilon(self):
        # 탐험 비율을 감쇠합니다.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save_model(self, model_dir):
        # 모델 디렉토리가 없으면 생성합니다.
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Q network 모델을 저장합니다.
        model_path = os.path.join(model_dir, 'q_network.h5')
        self.q_network.save(model_path)
        print(f'Saved Q network model to {model_path}')

        # target network 모델을 저장합니다.
        target_model_path = os.path.join(model_dir, 'target_network.h5')
        self.target_network.save(target_model_path)
        print(f'Saved target network model to {target_model_path}')
    def sav_reward(self):
        df = pd.DataFrame(self.reward_list, columns=['reward'])
        filename = 'model_5_reward'
        path = '/Users/white/Desktop/valak/Valak/Main_ValaK/DQN_project/rewards/' + filename + '.csv'
        if not os.path.isfile(path):  # 파일이 존재하지 않으면
            df.to_csv(path, index=False, mode='w')  # 파일을 새로 만들어서 저장하고
        else:  # 파일이 이미 존재하면
            df.to_csv(path, index=False, header=False, mode='a')  # 데이터만 추가해서 저장
        self.reward_list = []
    def show_reward(self):
        plt.figure(figsize=(10,5))
        plt.plot(self.reward_list, label='rewards')
        plt.legend(loc='upper left')
        plt.title('DQN')
        plt.show()
if __name__ == '__main__':
    directory = '/Users/white/Desktop/valak/Valak/Main_ValaK/DQN_project/filtered_data'
    file_names = os.listdir(directory)
    filename = file_names[0]
    path = '/Users/white/Desktop/valak/Valak/Main_ValaK/DQN_project/filtered_data/' + filename
    agent = DQNAgent(path)
    num = 0
    rand_list = []
    try:
        while num != len(file_names):
            random_int = random.randint(0, len(file_names)-1)
            if random_int in rand_list:
                continue
            else:
                rand_list.append(random_int)
                filename = file_names[random_int]
                path = '/Users/white/Desktop/valak/Valak/Main_ValaK/DQN_project/filtered_data/' + filename
                agent.change_env(path)
                agent.train()
                agent.save_model(model_dir='/Users/white/Desktop/valak/Valak/Main_ValaK/DQN_project/model_5')
                num+=1
                agent.show_reward()
                agent.sav_reward()
        #os.system('shutdown -s -f -t 0')
    except Exception as e:
        print(e)
        #os.system('shutdown -s -f -t 0')