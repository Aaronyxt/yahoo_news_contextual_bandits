# -*- coding: utf-8 -*-

import numpy as np
import random as rand
from policy.IndexPolicy import IndexPolicy
from scipy.linalg import hadamard

class CobraFHT(IndexPolicy):

    def __init__(self, nbArms, nbFeatures):
        self.nbArms = nbArms
        self.nbFeatures = nbFeatures
        self.alpha=0.1        
        
    def startGame(self, rounds, reductionDim):
        self.newFeatures = reductionDim
        self.X=np.matrix(np.eye(self.newFeatures))
        self.b=np.matrix(np.zeros(self.newFeatures))
        self.padColumn = int(np.power(2,np.ceil(np.log2(self.nbFeatures))) - self.nbFeatures) # let the input dimension be power of 2
        
        self.z=np.matrix(np.zeros(self.newFeatures))
        self.R=np.matrix(np.zeros((self.nbFeatures + self.padColumn, self.newFeatures)))
        self.IdentityMatrix =np.matrix(np.eye(self.nbFeatures + self.padColumn))        
        self.theta=np.linalg.inv(self.X)*np.transpose(np.matrix(self.b))        
        #self.theta = np.linalg.inv((np.matrix(np.eye(self.newFeatures))/self.newFeatures + self.X.transpose()*self.X))*self.X.transpose()*np.transpose(np.matrix(self.b))# ridge regression
        ## FHT based on Paul's work in AISTATS 2013
        H = np.matrix(hadamard(self.nbFeatures + self.padColumn))*np.sqrt(1/(self.nbFeatures + self.padColumn))
        D = np.matrix(np.zeros((self.nbFeatures + self.padColumn, self.nbFeatures + self.padColumn)))
        S = np.matrix(np.zeros((self.nbFeatures + self.padColumn, self.newFeatures)))   
        
        for i in range(self.nbFeatures + self.padColumn):
            randomValue = rand.random()
            if randomValue < 1/2.0:
                value = 1.0
            else:
                value = -1.0
            D.itemset(i, i, value)  
        for i in range(self.newFeatures):
            randomInt = np.random.randint(self.nbFeatures + self.padColumn)
            selectColumn = self.IdentityMatrix[:, randomInt]
            S[:, i] = selectColumn
            
        self.R = D*H*S*np.sqrt((self.nbFeatures + self.padColumn)/self.newFeatures)
        
        
    def getReward(self, armid, arm, reward):   
        self.inputArm = np.hstack((np.matrix(arm), np.matrix(np.zeros(self.padColumn))))
        self.z = (self.inputArm*self.R).transpose()
        self.X+=self.z*self.z.transpose()
        self.b+=np.dot(self.z.transpose(), reward)
        self.theta=np.linalg.inv(self.X)*np.transpose(np.matrix(self.b))
       #self.theta = np.linalg.inv((np.matrix(np.eye(self.newFeatures))/self.newFeatures + self.X.transpose()*self.X))*self.X.transpose()*np.transpose(np.matrix(self.b)) # ridge regression
    
    def computeIndex(self, armid, arm):
        """
        arm is the contextual information
        """
        self.inputArm = np.hstack((np.matrix(arm), np.matrix(np.zeros(self.padColumn))))
        self.z = (self.inputArm*self.R).transpose()
        result = (self.theta.transpose()*self.z).item(0,0)+self.alpha*np.sqrt((self.z.transpose()*np.linalg.inv(self.X)*self.z).item(0,0)) 
        return result