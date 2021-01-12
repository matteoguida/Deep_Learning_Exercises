"""
NEURAL NETWORKS AND DEEP LEARNING
PHYSICS OF DATA - Department of Physics and Astronomy
A.A. 2019/20 (6 CFU)
Dr. Alberto Testolin, Dr. Federico Chiariotti


Author: Matteo Guida


Description:

A 2D Maze is solved by a reinforcement learning algorithm. 

 - In the class Agent the Q-table is inizialized and a function to perform the Q-update and one to select and apply the behaviour policy are implemented.
 - The training function allows to train the agent for a given number of episodes, i.e. # of matches. In the end the solved maze can be plotted. 

"""



import numpy as np
import scipy.special as sp
import Environment
from tqdm import trange


class Agent:

    states = 1
    actions = 1
    discount = 0.9
    max_reward = 1
    qtable = np.matrix([1])
    softmax = False
    sarsa = False
    

    
    # initialize
    def __init__(self, states, actions, discount, max_reward, softmax, sarsa):
        self.states = states
        self.actions = actions
        self.discount = discount
        self.max_reward = max_reward
        self.softmax = softmax
        self.sarsa = sarsa
        # initialize Q table
        self.qtable = np.ones([states, actions], dtype = float) * max_reward / (1 - discount)
        
        
    # update function (Sarsa and Q-learning)
    def update(self, state, action, reward, next_state, alpha, epsilon):
        # find the next action (greedy for Q-learning, using the decision policy for Sarsa)
        next_action = self.select_action(next_state, epsilon)
        # calculate long-term reward with bootstrap method
        if (self.sarsa):
            observed = reward + self.discount * self.qtable[next_state, next_action]
        else:
            observed = reward + self.discount * np.max(self.qtable[next_state])
        # bootstrap update
        self.qtable[state, action] = self.qtable[state, action] * (1 - alpha) + observed * alpha

        
    # action policy: implements epsilon greedy and softmax
    def select_action(self, state, epsilon):
        qval = self.qtable[state]
        prob = []
        if (self.softmax):
            # use Softmax distribution
            prob = sp.softmax(qval/(epsilon+1e-4))
        else:
            # assign equal value to all actions
            prob = np.ones(self.actions) * epsilon / (self.actions - 1)
            # the best action is taken with probability 1 - epsilon
            prob[np.argmax(qval)] = 1 - epsilon
        return np.random.choice(range(0, self.actions), p = prob)

    
def training(agent, x, y, initial, goal, alpha_array, epsilon_array, walls=[], sands=[], softmax=False, sarsa=False, log=True,
             episodes=3000, discount=0.9, episode_length=50, verbose=False,plot_maze=[True,"solved_"]):

    reward_list = []
     
        
    for index in range(0, episodes):

        # start from a random state
        initial = [np.random.randint(0, x), np.random.randint(0, y)]
        while(initial in walls) or (np.array(initial) == np.array(goal)).all():
                initial = [np.random.randint(0, x), np.random.randint(0, y)]
        # initialize environment
        state = initial
        env = Environment.Environment(x = x, y = y, initial= state,goal =  goal, walls=walls, sands=sands)
        reward = 0

        # run episode
        for _ in range(0, episode_length):

            # find state index
            state_index = state[0] * y + state[1]
            # choose an action
            action = agent.select_action(state_index, epsilon_array[index])
            # the agent moves in the environment
            result = env.move(action)
            # Q-learning update
            next_index = result[0][0] * y + result[0][1]
            agent.update(state_index, action, result[1], next_index, alpha_array[index], epsilon_array[index])
            # update state and reward
            reward += result[1]
            state = result[0]


        reward /= episode_length    

        reward_list.append(reward)
        if verbose is True and index==(episodes-1):
            print("\nThe final mean reward per step is ora : ",reward)

    # plot the maze structure.        
    if plot_maze[0] is True:
        env.plot_maze(title="Structure",filename="./plots/maze_structure.png")
    
    # test network from the same starting point
    env = Environment.Environment(x=x, y=y, initial=initial, goal=goal, walls=walls, sands=sands, save_path=True)
    state = env.start
    for _ in range(0, episode_length):
        # find state index
        state_index = state[0] * y + state[1]
        # choose an action
        action = agent.select_action(state_index, epsilon_array[-1])
        # the agent moves in the environment
        result = env.move(action)
        # update state
        state = result[0]
    
    fname = 'test_result.png'
    if plot_maze[0] is True:
        env.plot_maze(iter=True,title="Solution", filename="./plots/"+plot_maze[1]+fname)
    
    return reward_list