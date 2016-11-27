# -*- coding: utf-8 -*-

'''clear all the variables in explorer'''
#import sys
#sys.modules[__name__].__dict__.clear()

import datetime # to show the date and time
import timeit # to calculate the time consumption
import random as rand

from environment.MAB import MAB
from numpy import *
from matplotlib.pyplot import *

from arm.Bernoulli import Bernoulli
from arm.Poisson import Poisson
from arm.Exponential import Exponential
from policy.UCB import UCB
from policy.klUCB import klUCB
from policy.RandChoice import RandChoice
from policy.epsilonGreedy import epsilonGreedy

from policy.SLUCB import SLUCB
from policy.BallExplore import BallExplore
from policy.CobraRS import CobraRS
from policy.CobraSG import CobraSG
from policy.CobraRD import CobraRD
from policy.CobraFHT import CobraFHT
from policy.LinUCB import LinUCB

from Evaluation import *
from prior.load_yahoo import load_yahoo
from prior.load_arms import load_arms

# figure setting
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'black', 'yellow', 'red']
markers = ['+','o','>','<','^','v','*','x'] 
graphic = 'yes'

# running setting
nbRep = 5 # 10
horizon = 100 #1000
reductionDim = 6

#datafile="simulation/ydata-fp-td-clicks-v1_20090501"
#data = load_yahoo(datafile)
fb = "simulation/arms"
data = load_arms(fb)
#print('#users: '+str(data.nbusers))
#print('#arms: '+str(data.nbarms))
#print('#reductionDim: '+str(reductionDim))
#print('#features: '+str(data.nbfeatures))
#print('#feedbacks: '+str(sum(len(data.user2arms.get(i)) for i in data.user2arms.keys())))
#print('#rounds: '+str(horizon))
#print('#Repetitions: '+str(nbRep))
trunc = 10
#print(np.array(data.user2arms.keys()))
data.user2arms[0] = 440
sampled_u = [0]

env = MAB(data.arm2features, {sampled_u[0]:data.user2arms[sampled_u[0]]})
K = env.nbArms
policies = [LinUCB(K,data.nbfeatures),UCB(env.nbArms),RandChoice(K,data.nbfeatures),epsilonGreedy(K,data.nbfeatures)]
 
tsav = int_(linspace(10,horizon-1,20))

if graphic == 'yes':
    figure(1)
    
#fileCumu = 'MAB_cumu_reward_out'+'#users'+str(data.nbusers)+'#arms'+str(data.nbarms)+'#features'+str(data.nbfeatures)+'#rounds'+str(horizon)+'#Repetitions'+str(nbRep)+'#reductionDim'+str(reductionDim)
#fileTime = 'Consuming_time'+'#users''#users'+str(data.nbusers)+'#arms'+str(data.nbarms)+'#features'+str(data.nbfeatures)+'#rounds'+str(horizon)+'#Repetitions'+str(nbRep)+'#reductionDim'+str(reductionDim)
#fileAllCumu = 'All_cum_reward'+'#users'+str(data.nbusers)+'#arms'+str(data.nbarms)+'#features'+str(data.nbfeatures)+'#rounds'+str(horizon)+'#Repetitions'+str(nbRep)+'#reductionDim'+str(reductionDim)
#
#outfile = open('results/'+fileCumu,'w')
#outfileTime = open('results/'+fileTime, 'w')
#outfileAllCumu = open('results/'+fileAllCumu, 'w')

k=0
for policy in policies:
    print(policy)
    print(datetime.datetime.now())
    timeBegin = timeit.default_timer()
    ev = Evaluation(env, policy, nbRep, horizon, reductionDim, tsav)
    timeEnd = timeit.default_timer()
    dataAllCumu = ev.allCumReward()
    print(datetime.datetime.now())
    print(ev.meanReward())
    
    cumuwards = np.array(ev.cumuwards())
#    outfile.write(str(policy)+'\n')
#    outfile.write(' '.join([str(x) for x in cumuwards]))
#    outfile.write('\n')
#    
#    outfileTime.write(str(policy)+'\n')
#    outfileTime.write(str(timeEnd - timeBegin))
#    outfileTime.write('\n')
    
#    outfileAllCumu.write(str(policy)+'\n')
#    for i in range(nbRep):
#        dataCumu = dataAllCumu[i, :]
#        outfileAllCumu.write(' '.join([str(x) for x in dataCumu]))
#        outfileAllCumu.write('\n')
    # plot figure
    if graphic == 'yes':
        ax = gca()
        semilogx(np.array(range(horizon)), cumuwards, color = colors[k], marker = markers[k])
        #loglog(np.array(range(horizon)), cumuwards, color = colors[k], marker = markers[k])
        xlabel('Rounds')
        ylabel('Cumulative Rewards')
        ax.set_ylim(ymax=horizon, ymin=-2)
        ax.set_xlim(xmax=horizon)
    k = k+1

#outfile.close()
#outfileTime.close()
#outfileAllCumu.close()

if graphic == 'yes':
    legend([policy.__class__.__name__ for policy in policies], loc=0)
    title('Cumulative rewards')
   # savefig('results/'+fileCumu+'.png',dpi = 500)
    show()
    

