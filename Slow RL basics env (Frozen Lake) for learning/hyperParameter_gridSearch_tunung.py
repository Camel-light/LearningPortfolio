import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False)

action_dict = {
    0: 'LEFT',
    1: 'DOWN',
    2: 'RIGHT',
    3: 'UP'
}

num_episodes = 50


def q_learning(lr, y, d, p):
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    total_rewards = np.zeros((num_episodes, 1))

    for i in range(num_episodes):
        epsilon = num_episodes / (i + d) ** p
        observation = env.reset()[0]
        total_reward = 0
        terminated = False

        while not terminated:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[observation])

            next_observation, reward, terminated, _, _ = env.step(action)

            if reward == 1:
                reward *= 2
            elif reward == 0 and terminated:
                reward = -0.8
            else:
                reward -= 0.001

            q_table[observation][action] = ((1 - lr) * q_table[observation][action]
                                            + lr * (reward + y * np.max(q_table[next_observation])))
            total_reward += reward
            observation = next_observation
        total_rewards[i] = total_reward

    avg_reward = np.mean(total_rewards)
    return avg_reward


# Grid search parameters
learning_rates = [0.1, 0.5, 0.9]
discount_factors = [0.8, 0.9, 0.99]
c_values = [num_episodes, num_episodes * 2, num_episodes * 3]
d_values = [1, 2, 3]
p_values = [2, 3, 4]

best_avg_reward = -np.inf
best_params = {}

for lr in learning_rates:
    for y in discount_factors:
        for d in d_values:
            for p in p_values:
                avg_reward = q_learning(lr, y, d, p)
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    best_params = {
                            'lr': lr,
                            'y': y,
                            'd': d,
                            'p': p
                        }

print(f"Best Parameters: {best_params}")
print(f"Best Average Reward: {best_avg_reward}")
