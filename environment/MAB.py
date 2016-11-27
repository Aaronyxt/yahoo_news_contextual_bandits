# -*- coding: utf-8 -*-
'''
Environement for a Multi-armed bandit problem with arms given in the 'arms' list 
'''

from Result import *
from environment.Environment import Environment
import numpy as np

class MAB(Environment):
    """Multi-armed bandit problem with arms given in the 'arms' list"""
    
    def __init__(self, arms,truth):
        if arms == None:
            self.arms = {}
            self.arms_id_dict = {}
            self.arm_counts = 0
            path_id = list(truth.keys())[0]
            self.nInstance = truth[path_id]-1
            self.arms = {}
            self.truth = {}
            path = "simulation/data"+"/"+str(path_id)+"/"
            for f in range(self.nInstance):
                count = 0
                with open(path+str(f), 'r') as file:
                    for data in file:
                        if count == 0:
                            article_id = data.strip()
                        if count == 1:
                            reward = int(data.strip())
                        if count > 1:
                            data = data.strip().split(" ")
                            if len(data) < 7:
                               continue
                            arm_id = data[0]
                            if arm_id not in self.arms_id_dict:
                                value_list = [0.0,0.0,0.0,0.0,0.0,0.0]
                                for d in range(6):
                                    value = data[d+1]
                                    info = value.split(":")
                                    
                                    value_list[int(info[0])-1] = float(info[1])    
                                self.arms_id_dict.update({arm_id:self.arm_counts})
                                self.arms.update({self.arm_counts:value_list})
                                self.arm_counts = self.arm_counts + 1
                            else:
                                continue
                        count = count + 1
                    if reward == 1:
                        reward_id = self.arms_id_dict[article_id]
                        if reward_id not in self.truth:
                            self.truth.update({reward_id:1})
            self.nbArms = len(self.arms)
        else:
            self.arms = arms
            self.nbArms = len(arms)
            self.truth=truth
        # supposed to have a property nbArms
    
    
    
    def play(self, policy, horizon, reductionDim):
        policy.startGame(horizon, reductionDim)
        result = Result(self.nbArms, horizon)
        for t in range(horizon):       
            if self.truth==None:
                choice = policy.choice()
#                if self.truth.has_key(c):
#                    reward=1
#                else:
#                    reward=0   
#                policy.getReward(choice, reward)
#                result.store(t, choice, reward)
                
                reward=self.arms[choice].draw()
                policy.getReward(choice, None, reward)
                result.store(t, choice, reward)
            else:
#                choice = policy.choicex(self.arms)
#                #print len(self.arms.keys())
#                #print len(self.truth.keys())
#                for c in choice:                
#                    if self.truth.has_key(c):
#                        reward=1
#                    else:
#                        reward=0     
#                f_c=np.random.choice(choice)
#                policy.getReward(f_c, self.arms[f_c], reward)
#                result.store(t, f_c, reward)   
                choice = policy.choicex(self.arms)
                
				#print len(self.arms.keys())
				#print len(self.truth.keys())
                
#==============================================================================
#                 for c in choice:                
#                     if c in self.truth:
#                         reward=1
#                     else:
#                         reward=0     
#==============================================================================
                f_c = np.random.choice(choice)
                if f_c in self.truth:
                    reward = self.truth[f_c] # for ranking else reward = 1
                else:
                    reward = 0 
                policy.getReward(f_c, self.arms[f_c], reward)
                result.store(t, f_c, reward)
        return result
   
   
   