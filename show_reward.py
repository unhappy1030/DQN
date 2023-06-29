import csv
import matplotlib.pyplot as plt

def plot_rewards_from_csv(csv_file):
    reward_list = []

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row if present

        for row in reader:
            reward = float(row[0])  # Assuming reward values are in the first column
            reward_list.append(reward)

    plt.figure(figsize=(10, 5))
    plt.plot(reward_list, label='rewards')
    plt.legend(loc='upper left')
    plt.title('DQN')
    plt.show()

# Usage example
filename = 'model_reward'
path = '/Users/white/Desktop/valak/Valak/Main_ValaK/DQN_project/rewards/' + filename + '.csv'
plot_rewards_from_csv(path)