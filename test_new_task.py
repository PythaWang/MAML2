import numpy as np
from env.cartpole_task_list import envs_dict
from util.sutton_tile_coder import TileCoder
from agent.LinearAgent import SARSA
from meta import MAML
from maml2 import *
import copy
from interaction import run
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# generate data
# N = 4
# trials = 10
# trial_record = []
# init_param = np.ones((10, 10, 4)) * 0.5
#
# for trial in range(trials):
#     step_record = []
#     tasks_list = list(np.random.choice(list(envs_dict.keys()), N, replace=False))
#     p = copy.deepcopy(init_param)
#     params_list = None
#     for i in range(50):
#         c, params_list = classify(tasks_list, N, params_list)
#         p, p_cat, _ = MAML2(c, p, envs_dict, 10, 10, N)
#         env = envs_dict[np.random.randint(len(envs_dict))]
#         agent = Agent(env)
#         agent.Q_value = copy.deepcopy(p)
#         s1, _, _ = run(agent, env, True, 1, False)
#         minimum = np.argmin([np.linalg.norm(agent.Q_value - p_cat[i]) - np.linalg.norm(p - p_cat[i])
#                              for i in range(N)])
#         agent.Q_value = copy.deepcopy(p_cat[minimum])
#         s2, _, mu = run(agent, env, True, 1999, False)
#         # print('pre', steps[:10])
#         # print('past', steps[-10:])
#         tasks_list.append(env.id)
#         params_list.append(copy.deepcopy(agent.Q_value) * mu)
#         step_record.append(np.mean((s1+s2)[:10]))

#         # p, _ = MAML(tasks_list, p, envs_dict, 100)
#         # env = envs_dict[np.random.randint(len(envs_dict))]
#         # agent = Agent(env)
#         # agent.Q_value = copy.deepcopy(p)
#         # steps, _, mu = run(agent, env, True, 2000, False)
#         # # print('pre', steps[:10])
#         # # print('past', steps[-10:])
#         # tasks_list.append(env.id)
#         # step_record.append(np.mean(steps[:10]))
#     # plt.plot(step_record)
#     # plt.show()
#     trial_record.append(step_record)
# np.save('result/test_sequential_task_maml2_4', trial_record)

# plot
data = pd.DataFrame()
trial_record = {}
trial_record[0] = np.load('result/test_sequential_task_maml.npy')
trial_record[1] = np.load('result/test_sequential_task_maml2_2.npy')
trial_record[2] = np.load('result/test_sequential_task_maml2.npy')
trial_record[3] = np.load('result/test_sequential_task_maml2_4.npy')
algs = ['MAML', 'MAML$^2$(N=2)', 'MAML$^2$(N=3)', 'MAML$^2$(N=4)']
for i, alg in enumerate(algs):
    record = trial_record[i]
    for trial_num in range(1, 2):
        trial = record[trial_num]
        for task_num, steps in enumerate(trial):
            data = data.append(pd.DataFrame([[alg, trial_num, task_num, steps]],
                               columns=['algorithm', 'trial_num', 'task_num', 'step']), ignore_index=True)

plt.figure(figsize=(12, 6))
sns.set(font='Times New Roman',font_scale=2)
p = sns.lineplot(data=data, x='task_num', y='step', hue='algorithm')
p.set_xlabel('Task number')
p.set_ylabel('Step')
plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
plt.tight_layout()
plt.savefig('picture/sequential_test.png')




