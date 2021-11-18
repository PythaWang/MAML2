import numpy as np
from sklearn.cluster import KMeans


def k_means(measure: dict, task_classification: dict):
    label = ['easy', 'medium', 'hard']
    data = [v for v in measure.values()]
    data = np.array(data).reshape(-1, 1)
    estimator = KMeans(n_clusters=3, random_state=1)
    estimator.fit(data)
    labels = estimator.labels_
    clusters_ = dict()
    for i in range(3):
        cluster = data[labels == i]
        if not len(cluster) == 0:
            center = min(cluster)
            clusters_[center[0]] = cluster
    keys = sorted(clusters_.keys())
    # task_classification_ {cluster_description : measure_list}
    task_classification_ = dict()
    for i in range(len(keys)):
        task_classification_[label[i]] = clusters_[keys[i]]
    new_dict = dict()
    for k, v in measure.items():
        if v in new_dict.keys():
            new_dict[v].append(k)
        else:
            new_dict[v] = [k]
    for k, v in task_classification_.items():
        task_list = []
        for m in v:
            if m[0] in new_dict.keys():
                for i in range(len(new_dict[m[0]])):
                    task_list.append(new_dict[m[0]][i] )
                new_dict.pop(m[0])
        task_classification[k] = task_list
    return id_to_cluster(task_classification)


def k_means_for_param(params_array, N):
    size = len(params_array)
    data = params_array.reshape(size, -1)
    estimator = KMeans(n_clusters=N, random_state=1)
    estimator.fit(data)
    return estimator.labels_, estimator.cluster_centers_


def id_to_cluster(task_classification):
    id_to_cluster_dict = dict()
    for k, v in task_classification.items():
        for i in range(len(v)):
            id_to_cluster_dict[v[i]] = k
    return id_to_cluster_dict
