import gymnasium as gym
import numpy as np
from time import sleep
import matplotlib.pyplot as plt

# Create environment
env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode='human')

# Slow with explanations
verbose_mode = True
speed = 1 if verbose_mode else 0

action_dict = {0: 'LEFT', 1: 'DOWN', 2: 'RIGHT', 3: 'UP'}

# Initialize the Q-table with zeros
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Initialize a secondary table to keep track of updates
updates_table = np.zeros((env.observation_space.n, env.action_space.n))

# Set the learning rate and discount factor
lr = 1  # If deterministic  lr=1 is optimal
y = 0.88

# Number of episodes to run
num_episodes = 50

# Store rewards for all episodes
total_rewards = np.zeros((num_episodes, 1))

# Parameters for polynomial decay
d = 1  # Prevents division by zero and can be used to adjust the initial value
p = 3  # Higher make epsilon decay faster => exploitation sooner

# Run Q-learning
for i in range(num_episodes):
    epsilon = num_episodes / (i + d) ** p
    # Reset the env, put agent to initial position 0
    observation = env.reset()[0]

    # Initialize variables for the current episode
    total_reward = 0
    terminated = False

    # Run episode
    while not terminated:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
            print("Chose random action")
        else:
            action = np.argmax(q_table[observation])
            print("Chose greedy action")

        if verbose_mode:
            print("Next action: ", action_dict[action])
            sleep(speed)

        next_observation, reward, terminated, truncated, info = env.step(action)
        if reward == 1:
            reward *= 2  # More weight goal
        elif reward == 0 and terminated is True:
            reward = -0.8
        else:
            reward -= 0.005  # Penalty weight for moving

        if verbose_mode:
            print("Position: ", next_observation, "Reward: ", reward, "Terminated: ", terminated)

        # Update the Q-table
        q_table[observation][action] = ((1 - lr) * q_table[observation][action]
                                        + lr * (reward + y * np.max(q_table[next_observation])))

        # Mark the updated value in the updates_table
        updates_table[observation][action] = 1

        # Update the total reward and the current state (position)
        total_reward += reward
        observation = next_observation

    # Print the Q-table at the end of the episode using updates_table for highlighting
    print("\n", "State/Action |", " | ".join([f"{action:^7}" for action in action_dict.values()]), "|")
    print("-" * 65)

    for k, state_values in enumerate(q_table):
        state_actions_list = []
        for l, value in enumerate(state_values):
            if updates_table[k][l] == 1:  # Check if the value was updated
                state_actions_list.append(f"\033[41m{value:7.2f}\033[0m")  # Highlight in dark red
            else:
                state_actions_list.append(f"{value:7.2f}")

        state_actions = " | ".join(state_actions_list)
        print(f"State {k:02}     |", state_actions, "|")
    print("\n")
    # Reset updates_table for next episode
    updates_table.fill(0)

    total_rewards[i] = total_reward


env.close()  # Close the environment

cumsum = np.cumsum(total_rewards)
avg_rewards = np.zeros((cumsum.shape))
for j in range(num_episodes):
    avg_rewards[j] = cumsum[j] / (j + 1)

plt.figure()
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Training Performance')
plt.plot(np.arange(num_episodes), avg_rewards)
plt.show()