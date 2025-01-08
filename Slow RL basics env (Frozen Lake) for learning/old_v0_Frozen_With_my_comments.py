Q = np.zeros([env.observation_space.n,env.action_space.n])  # Initialize table with all zeros

# Set learning parameters
lr = .8  # Learning rate - how much of the Q-value estimate we adopt during update.
y = .95  # Discount factor
num_episodes = 2000 # episodes for training
#create lists to contain total rewards and steps per episode
#jList = []
rList = []  # Rewards achieved in each episode
for i in range(num_episodes):  #  Each time the agent interacts with the environment and updates the Q-table based on its experiences.
    
    s = env.reset()   # Reset environment and get initial state
    rAll = 0  # To keep track of total reward for the episode
    d = False # 'd' stands for 'done', indicating whether the episode has finished.
    j = 0     # Counter for steps in each episode

    #The Q-Table learning algorithm for 99 steps or d == True
    while j < 99:
        j+=1
        #Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))  # noise decreases with progress (1./(i+1), making the agent exploit more as it learns.
        
        # Take action a => get new state s1, reward r, and done d, from environment
        s1,r,d,_ = env.step(a)
        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a]) #  Take maximizing action for s1
        #  Update reward and state
        rAll += r
        s = s1
        if d == True:
            break
    #jList.append(j)
    rList.append(rAll)