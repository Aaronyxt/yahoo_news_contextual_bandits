# -*- coding: utf-8 -*-


from numpy import *
from matplotlib.pyplot import *
import matplotlib.pyplot as plt

colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'black']
markers = ['^','*','+','o','>','<','x'] 

names = ['Cobra','LinUCB','LinRel','klUCB','BayesUCB']
data = np.loadtxt('MAB_cumu_out')

horizon = len(data[1])
tsav = int_(linspace(10,horizon-1,20))

policy_number = len(data)
print(policy_number)

figure(1)
for policy in range(policy_number):
    ax = gca()
    cumuwards = data[policy]
    xx = np.array(range(1,int(horizon/10)+1))*10
    yy = cumuwards[xx-1]
    yy_error = []
    for pos in xx:
        all_xx = cumuwards[range(pos)]
        yy_error.append(np.std(all_xx))
    plot(np.array(range(horizon)), cumuwards, color = colors[policy], marker = markers[policy])
    plt.errorbar(xx, yy, yerr=yy_error, linestyle="None", color = colors[policy], marker = markers[policy])
    #loglog(np.array(range(horizon)), cumuwards, color = colors[k], marker = markers[k])
    xlabel('Rounds')
    ylabel('Rewards')
    ax.set_ylim(ymax=100,ymin=-1)
    ax.set_xlim(xmax=55)

legend([policy_name for policy_name in names], loc=0)
title('Cumulative rewards')
savefig('cumulative_rewards.png',dpi = 500)