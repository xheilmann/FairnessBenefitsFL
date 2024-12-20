import sys, os



working_dir = '.'
sys.path.insert(1, os.path.join(working_dir, 'FedFB'))
os.environ["PYTHONPATH"] = os.path.join(working_dir, 'FedFB')

from DP_run import *

sim_dp('fedfb', 'multilayer perceptron', 'states_income_sex')

'''
lr=[.001, .005, .01]
alpha=[.001, .05]
for a in alpha :
    for b in lr:
        print(a,b)
        sim_dp_man('fedfb', 'multilayer perceptron', 'states_income_sex', alpha=a, learning_rate=b, num_sim=1)
'''