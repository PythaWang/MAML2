import pandas as pd
import numpy as np

# alg = [('new_task_test_maml', 'MAML'), ('new_task_test_maml2', 'MAML$^2$')]
# alg = [('maml_3000', 'MAML'), ('maml2_new_100_10', 'MAML$^2$')]
alg = [('maml2', 'MAML$^2$'), ('maml2_no_reset', 'MAML$^2$(no_reset)')]
path = '../result/'
compare_task = [[0, 1], [2, 3, 4, 5, 6, 7, 8], [9]]

data = pd.DataFrame()
for a in alg:
    record = np.load(path+a[0]+'.npz')
    for c, tasks in enumerate(compare_task):
        for i in tasks:
            task = record['task_{}'.format(i)]  # fetch a record of a task
            for trial in range(len(task)):  # fetch a record of a trail in the task
                for e, s in enumerate(task[trial]):  # fetch an episode data in the trail
                    data = data.append(pd.DataFrame([[a[1], i, trial, e, s, c]],
                                       columns=['algorithm', 'task', 'trial', 'episode', 'step', 'category']),
                                       ignore_index=True)

for c, tasks in enumerate(compare_task):
    for i in tasks:
        task = np.load(path + 'task_{}.npy'.format(i))  # fetch a record of a task
        for trial in range(len(task)):  # fetch a record of a trail in the task
            for e, s in enumerate(task[trial]):  # fetch an episode data in the trail
                data = data.append(pd.DataFrame([['LFS', i, trial, e, s, c]],
                                   columns=['algorithm', 'task', 'trial', 'episode', 'step', 'category']),
                                   ignore_index=True)

data.to_csv('../result/{}_vs_{}'.format(alg[0][0], alg[1][0]))

