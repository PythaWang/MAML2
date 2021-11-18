from util.heatmap import show_in_heatmap_for_params
import numpy as np



p_maml = np.load('../result/maml_param_meta.npy')
p_maml2 = np.load('../result/maml2_param_meta.npy')
cat = np.load('../result/maml2_param_cat.npz')
cat0 = sum(cat['cat0']) / len(cat['cat0'])
cat1 = sum(cat['cat1']) / len(cat['cat1'])
cat2 = sum(cat['cat2']) / len(cat['cat2'])
p1 = sum(p_maml) / len(p_maml)
p2 = sum(p_maml2) / len(p_maml2)
param_cat = [cat0, cat1, cat2]
show_in_heatmap_for_params([p1, p2, cat2, cat0, cat1], ['MAML', 'MAML$^2$', 'category 0', 'category 1', 'category 2'])
