import numpy as np
from agent.Agent import Agent
from env.gridword_task_list import envs_dict
import copy
from interaction import run


BETA = 1
delta_converge = 0.00001

def MAML(c, cat_param, envs_list: dict, max_iteration=100):
    losses_list = []
    iteration = 0
    while True:
        iteration += 1
        id_sample = np.random.choice(c)
        cat_param_ = copy.deepcopy(cat_param)
        env = envs_list[id_sample]
        agent = Agent(env=env)
        agent.Q_value = copy.deepcopy(cat_param)  # initialization by category parameter
        steps_list, _, _ = run(agent, env, True, 5, False)  # get param' by update

        # update meta-parameter (category parameter)
        cat_param += BETA/len(c) * (agent.Q_value - cat_param)  # update category parameter by param'
        loss = np.linalg.norm(cat_param_ - cat_param)
        losses_list.append(loss)
        if loss < delta_converge or iteration >= max_iteration:
            break
    return cat_param, losses_list

if __name__ == '__main__':
    rand = 50
    max_iteration = 3000
    p = []
    c = list(envs_dict.keys())
    record = {i: [] for i in range(len(envs_dict))}
    for r in range(rand):
        param_meta, ll = MAML(c, np.ones((10,10,4)) * 0.5, envs_dict, max_iteration)
        for i in c:
            agent = Agent(env=envs_dict[i])
            agent.Q_value = copy.deepcopy(param_meta)
            t, _, _ = run(agent, envs_dict[i], True, 10, False)
            record[i].append(t)
            print(i, t)
        p.append(param_meta)
    np.save('result/maml_param_meta', p)
    np.savez('result/maml_{}'.format(max_iteration), **{'task_{}'.format(i): record[i] for i in range(len(envs_dict))})