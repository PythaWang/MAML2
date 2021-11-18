MAX_t = 1000


def run(agent, env, train: bool, max_episodes, render: True):
    e = 0
    steps_list = []
    losses_list = []
    dataset = []
    mu = np.zeros(shape=agent.Q_value.shape)
    t = 0
    while e < max_episodes:
        mu = np.zeros(shape=agent.Q_value.shape)
        loss_mean = 0
        path = []
        env.link_record(agent.Q_value, path)  # for render function in Env to show q_value and path in render
        state = env.reset()
        path.append(state)
        # choose an action
        action = agent.choose_action(state)
        t = 0
        while True:
            # step
            mu[state % agent.width][state // agent.width][action] += 1
            next_s, reward, done = env.step(action)
            path.append(next_s)
            if e == 0 and render:
                env.render()
            t += 1
            next_a = agent.choose_action(next_s)
            # update the policy

            if train:
                loss = agent.update_policy(state, action, reward, next_s, next_a, done)
                loss_mean = loss_mean + (loss - loss_mean) / t
            else:
                dataset.append((state, action, reward, next_s, next_a, done))
            state = next_s
            action = next_a
            if done or t >= MAX_t:
                steps_list.append(t)
                if train:
                    losses_list.append(loss_mean)
                e += 1
                break
    if train:
        return steps_list, losses_list, mu/t
    else:
        return dataset


if __name__ == '__main__':
    from agent.Agent import Agent
    from env.gridword_task_list import envs_dict
    import numpy as np

    task_id = 9
    step_record = []
    for r in range(50):
        env = envs_dict[task_id]
        agent = Agent(env)
        steps, _, _ = run(agent, env, True, 10, False)
        step_record.append(steps)
    np.save('result/task_{}'.format(task_id), step_record)
    # import matplotlib.pyplot as plt
    # plt.plot(steps)
    # plt.show()
