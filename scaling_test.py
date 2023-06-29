import matplotlib.pyplot as plt

def scale_reward(reward):
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

# 스케일링된 reward 값 계산

x = []
y = []
for i in range(600):
    reward = -0.3 + i * 0.001
    scaled_reward = scale_reward(reward)
    x.append(reward)
    y.append(scaled_reward)

plt.plot(x, y)
plt.xlabel('Reward')
plt.ylabel('Scaled Reward')
plt.title('Reward Scaling')
plt.show()