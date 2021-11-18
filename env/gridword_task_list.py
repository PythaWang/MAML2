from env.GridWord import Env

# generate the task space
envs_dict = dict()
blocks = [2, 12, 29, 38, 43, 44, 45, 46, 47, 52, 61, 62, 67, 68, 76, 81, 86, 96]
envs_dict[0] = Env(0, 10, 0, 0, 39, blocks)
envs_dict[1] = Env(1, 10, 0, 0, 48, blocks)
envs_dict[2] = Env(2, 10, 0, 0, 69, blocks)
envs_dict[3] = Env(3, 10, 0, 0, 79, blocks)
envs_dict[4] = Env(4, 10, 0, 0, 78, blocks)
envs_dict[5] = Env(5, 10, 0, 0, 77, blocks)
envs_dict[6] = Env(6, 10, 0, 0, 99, blocks)
envs_dict[7] = Env(7, 10, 0, 0, 98, blocks)
envs_dict[8] = Env(8, 10, 0, 0, 97, blocks)
envs_dict[9] = Env(9, 10, 0, 0, 90, blocks)