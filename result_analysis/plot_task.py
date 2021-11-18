import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

compare_task = [[0, 1], [2, 3, 4, 5, 6, 7, 8], [9]]
setting = 'maml_vs_maml2'
# setting = 'maml2_new_10_10_vs_maml2_new_no_reset_10_10'
path = '../result/' + setting
# category = [[0,1], [2,3,4,5,6,7,8], [9]]
data = pd.read_csv(path)
sns.set(font='Times New Roman', font_scale=2)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for j, t in enumerate(compare_task):
    palette = sns.color_palette("deep", 3)
    d = data[data['task']==t[0]]
    for i in t[1:]:
        d = d.append(data[data['task']==i])
    p = sns.lineplot(data=d, x='episode', y='step', hue='algorithm', palette=palette, ax=axes[j])
    p.set_xlabel('Episode')
    p.set_ylabel('Step')
    p.set_title('category {}'.format(j))
    p.set_yticks([0, 200, 400, 600, 800, 1000])
    axes[j].legend(loc=1)
fig.tight_layout()

plt.savefig('../picture/{}.png'.format(setting))
plt.show()

