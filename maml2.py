import copy

import numpy as np

from env.GridWord import Env
from agent.Agent import Agent
from interaction import run
import matplotlib.pyplot as plt
from util.k_means import k_means_for_param
from meta import MAML
from util.heatmap import show_in_heatmap_for_params
from env.gridword_task_list import envs_dict


def classify(id_list, N, params_list=None):
    c = {i: [] for i in range(N)}
    if params_list is None:
        params_list = []
        for i in id_list:
            env = envs_dict[i]
            agent = Agent(env)
            steps_list, losses_list, mu = run(agent, env, True, 2000, False)
            params_list.append(copy.deepcopy(agent.Q_value) * mu)

    # cluster
    params_array = np.array(params_list)
    labels_list, centers_list = k_means_for_param(params_array, N)
    print("task:", id_list)
    print("label:", labels_list)

    for i, label in enumerate(labels_list):
        c[label].append(id_list[i])
    print(c)
    return c, params_list

def MAML2(c, param_meta, envs_list, max_iteration, inner_max_iteration, N):
    losses_list = []
    for iteration in range(max_iteration):
        param_meta_ = copy.deepcopy(param_meta)
        param_cat = {}
        for i in range(N):
            param_cat[i], losses_list_MAML = MAML(c[i], copy.deepcopy(param_meta), envs_list, inner_max_iteration)
            param_meta += 1 / N * (param_cat[i] - param_meta)
        losses_list.append(np.linalg.norm(param_meta_ - param_meta))
    return param_meta, param_cat, losses_list


if __name__ == '__main__':
    for inner in [1, 5, 10]:
        N = 3
        rand = 50
        max_iteration = int(200/inner)
        inner_max_iteration = inner

        record = {i: [] for i in range(len(envs_dict))}
        meta_record = []
        cat_record = {i: [] for i in range(N)}

        # the first way to implement (for testing)
        # for r in range(rand):
        #     param_cat = {i: [] for i in range(N)}
        #     np.random.seed(r)
        #     param_meta = np.ones((10, 10, 4)) * 0.5
        #
        #     # classification
        #     c, params_list = classify(list(envs_dict.keys()), N)
        #
        #     # MAML
        #     for iteration in range(max_iteration):
        #         for i in range(N):
        #             param_cat[i], losses_list_MAML = MAML(c[i], copy.deepcopy(param_meta), envs_dict, inner_max_iteration)
        #             param_meta += 1/N * (param_cat[i] - param_meta)
        #
        #     tasks_list = []
        #     for i in range(len(envs_dict)):
        #         env = envs_dict[i]
        #         tasks_list.append(env.id)
        #         agent = Agent(env=env)
        #         agent.Q_value = copy.deepcopy(param_meta)
        #         s1, l, _ = run(agent, env, True, 1, False)
        #         render = False
        #
        #         minimum = np.argmin([np.linalg.norm(agent.Q_value - param_cat[i]) - np.linalg.norm(param_meta - param_cat[i])
        #                    for i in range(N)])
        #         print("(category, task):", minimum, i)
        #
        #         agent.Q_value = copy.deepcopy(param_cat[minimum])
        #         s2, _, mu = run(agent, env, True, 1, render)
        #         record[env.id].append(s1+s2)
        #     meta_record.append(param_meta)
        #     for i in range(N):
        #         cat_record[i].append(param_cat[i])
        # np.save('result/maml2_param_meta_past', meta_record)
        # np.savez('result/maml2_param_cat_past', **{'cat{}'.format(i): cat_record[i] for i in range(N)})
        # np.savez('result/maml2_{}_{}'.format(max_iteration, inner_max_iteration), **{'task_{}'.format(i): record[i] for i in range(len(envs_dict))})

        # the second way to implement (call functions classify and MAML2)
        tasks_list = list(envs_dict.keys())
        init_param = np.ones((10, 10, 4)) * 0.5

        for r in range(rand):
            p = copy.deepcopy(init_param)
            params_list = None
            c, params_list = classify(tasks_list, N, params_list)
            p, p_cat, _ = MAML2(c, p, envs_dict, max_iteration, inner_max_iteration, N)
            for i in range(len(envs_dict)):
                env = envs_dict[i]
                agent = Agent(env)
                agent.Q_value = copy.deepcopy(p)
                s1, _, _ = run(agent, env, True, 1, False)
                minimum = np.argmin([np.linalg.norm(agent.Q_value - p_cat[i]) - np.linalg.norm(p - p_cat[i])
                                     for i in range(N)])
                agent.Q_value = copy.deepcopy(p_cat[minimum])
                s2, _, mu = run(agent, env, True, 9, False)
                print(i, s1, s2)
                record[i].append(s1+s2)
            meta_record.append(p)
            for i in range(N):
                cat_record[i].append(p_cat[i])
        np.save('result/maml2_param_meta_new', meta_record)
        np.savez('result/maml2_param_cat_new', **{'cat{}'.format(i): cat_record[i] for i in range(N)})
        np.savez('result/maml2_new_{}_{}'.format(max_iteration, inner_max_iteration), **{'task_{}'.format(i): record[i] for i in range(len(envs_dict))})