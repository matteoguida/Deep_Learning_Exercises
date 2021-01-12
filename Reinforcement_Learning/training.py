"""
NEURAL NETWORKS AND DEEP LEARNING
PHYSICS OF DATA - Department of Physics and Astronomy
A.A. 2019/20 (6 CFU)
Dr. Alberto Testolin, Dr. Federico Chiariotti


Author: Matteo Guida

Description:

A 2D Maze is solved by a reinforcement learning algorithm. 

 - In the following script the results are load in order to  meet  the  request  in  the  assignment. 
 Please  EX5_Matteo_Guida.ipynb look at the jupyter notebook in order to have aclearer view of the operations carried out.

"""



if __name__ == "__main__":

    import pickle
    import matplotlib.pyplot as plt

with open('./results/linear_schedule_rewards.data', "rb") as input_file:
    linear_load_rewards = pickle.load(input_file)
with open('./results/exp_schedule_rewards.data', "rb") as input_file:
    exp_load_rewards = pickle.load(input_file)

plot_titles = [r"Q-learning $\epsilon$-greedy",r"Q-learning Softmax",r"SARSA $\epsilon$-greedy"
               ,r"SARSA Softmax"]

exp_list_join= list(zip(exp_load_rewards,linear_load_rewards,plot_titles))

print(" \n The script only load the results obtained averaging 50 iterations. \n The code can be found in EX5_Matteo_Guida.ipynb.")

fig, ax = plt.subplots(1,4, figsize=(22,4))

for i,list_rew in enumerate(exp_list_join):
    ax[i].scatter(range(len(list_rew[0].flatten())),list_rew[0].flatten(),color='r',s=4,alpha=0.2,label=r"Exponential $\epsilon$ Schedule")
    ax[i].scatter(range(len(list_rew[1].flatten())),list_rew[1].flatten(),color='b',s=4,alpha=0.2,label=r"Linear $\epsilon$ Schedule")
    ax[i].set_title(list_rew[2]+'\n Average Reward per Episode' ,fontsize=18)
    ax[i].set_xlabel("Episode Number [#]",fontsize=18)
    ax[i].tick_params(labelsize=15)
    ax[i].grid(alpha=0.2)
    ax[i].legend(fontsize=12)
    ax[i].set_ylim([min(list_rew[0].flatten()), 1])
ax[0].set_ylabel("Average Reward",fontsize=18)
    
plt.savefig('./plots/reward.png',bbox_inches="tight")
plt.show() 

