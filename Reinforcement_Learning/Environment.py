"""
NEURAL NETWORKS AND DEEP LEARNING
PHYSICS OF DATA - Department of Physics and Astronomy
A.A. 2019/20 (6 CFU)
Dr. Alberto Testolin, Dr. Federico Chiariotti


Author: Matteo Guida
Date of creation : 12/10/2020


Description:

A 2D Maze is solved by a reinforcement learning algorithm. 

 - In the class Environment four types of states are included: wall, sand, normal cell and goal. 
 A function to move the environment according to an action is implemented. After three function that can check if a state is one of the four types
 a function pure devoted to plotting purpose is considered. 

"""


from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt



class Environment:
    
    state = []
    goal = []
    boundary = []
    action_map = {
        0: [0, 0],
        1: [0, 1],
        2: [0, -1],
        3: [1, 0],
        4: [-1, 0],
    }
    
    def __init__(self, x, y, initial, goal, walls=[], sands=[], save_path=False):
        self.boundary = np.asarray([x, y])
        self.state = np.asarray(initial)
        self.goal = goal
        self.walls = walls
        self.sands = sands
        self.start = list(initial)
        self.state = np.asarray(self.start)
        self.save_path = save_path
        if self.save_path == True:
            self.followed_path = [self.start]
    
    def move(self, action):
        # start by default move.
        reward = 0
        movement = self.action_map[action]
        
        # check if it is a goal.
        if (action == 0 and (self.state == self.goal).all()):
            reward = 1
        next_state = self.state + np.asarray(movement)
        # check if it is a border or a wall. 
        if(self.check_boundaries(next_state) or self.check_wall(next_state)):
            reward = -1
        # check if it is a sand.
        elif(self.check_sand(next_state)):
            reward = -0.60
            self.state = next_state
            # save the history to reconstruct the followed path.
            if self.save_path:
                self.followed_path.append(list(next_state))
        else:
            self.state = next_state
            if self.save_path:
                self.followed_path.append(list(next_state))
        return [self.state, reward]

    def check_boundaries(self, state):
        out = len([num for num in state if num < 0])
        out += len([num for num in (self.boundary - np.asarray(state)) if num <= 0])
        return out > 0
    
    
    def check_wall(self, state):
        return list(state) in self.walls

    def check_sand(self, state):
        return list(state) in self.sands
    
    def plot_maze(self, title, iter=None, filename=None):
        plt.close('all')
        
        # Color the different cells in the maze. 
        maze_structure = np.zeros(self.boundary)
        maze_structure[self.goal[0], self.goal[1]] = 3
        for x_y in self.walls:
            maze_structure[x_y[0],x_y[1]] = 1
        for x_y in self.sands:
            maze_structure[x_y[0],x_y[1]] = 2
          
        cmap = colors.ListedColormap(['white','black','moccasin','green'])

        img = plt.imshow(maze_structure, cmap=cmap)
        if (iter is not None and self.save_path==True):
            iter_color = 'crimson'
            plt.plot(np.asarray(self.followed_path)[:,1],np.asarray(self.followed_path)[:,0], '--', linewidth=2, color=iter_color,label="Followed Path")
            plt.plot(np.asarray(self.followed_path)[0,1],np.asarray(self.followed_path)[0,0], marker=">", markersize=15, color=iter_color, label="Start")
            plt.plot(np.asarray(self.followed_path)[-1,1],np.asarray(self.followed_path)[-1,0], marker="X", markersize=15, color=iter_color, label="Goal")
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=15 )

        plt.axes().set_aspect('equal')
        plt.title('Maze '+title,fontsize=16)
        plt.tick_params(labelsize=14)
        plt.xticks(range(10))
        plt.yticks(range(10))
        plt.axes().invert_yaxis()
        if filename is not None:
            plt.savefig(filename)
        plt.show()