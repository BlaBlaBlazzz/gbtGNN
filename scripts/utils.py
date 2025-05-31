import os

import dgl
from dgl.data import citation_graph as citegrh, TUDataset
import torch as th
from dgl import DGLGraph
import numpy as np
import pandas as pd
import networkx as nx

from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity
import itertools
from sklearn.preprocessing import OneHotEncoder as OHE
import random
import json
import warnings

from catboost import CatBoostClassifier, CatBoostRegressor 


def load_cora_data():
    data = citegrh.load_cora()
    features = th.FloatTensor(data.features)
    labels = th.LongTensor(data.labels)
    train_mask = th.BoolTensor(data.train_mask)
    test_mask = th.BoolTensor(data.test_mask)
    g = DGLGraph(data.graph)
    return g, features, labels, train_mask, test_mask

def get_degree_features(graph):
    return graph.out_degrees().unsqueeze(-1).numpy()

def get_categorical_features(features):
    return np.argmax(features, axis=-1).unsqueeze(dim=1).numpy()

def get_random_int_features(shape, num_categories=100):
    return np.random.randint(0, num_categories, size=shape)

def get_random_norm_features(shape):
    return np.random.normal(size=shape)

def get_random_uniform_features(shape):
    return np.random.unifor(-1, 1, size=shape)

def merge_features(*args):
    return np.hstack(args)

def get_train_data(graph, features, num_random_features=10, num_random_categories=100):
    return merge_features(
        get_categorical_features(features),
        get_degree_features(graph),
        get_random_int_features(shape=(features.shape[0], num_random_features), num_categories=num_random_categories),
    )


def save_folds(dataset_name, n_splits=3):
    dataset = TUDataset(dataset_name)
    i = 0
    kfold = KFold(n_splits=n_splits, shuffle=True)
    dir_name = f'kfold_{dataset_name}'
    for trix, teix in kfold.split(range(len(dataset))):
        os.makedirs(f'{dir_name}/fold{i}', exist_ok=True)
        np.savetxt(f'{dir_name}/fold{i}/train.idx', trix, fmt='%i')
        np.savetxt(f'{dir_name}/fold{i}/test.idx', teix, fmt='%i')
        i += 1


def graph_to_node_label(graphs, labels):
    targets = np.array(list(itertools.chain(*[[labels[i]] * graphs[i].number_of_nodes() for i in range(len(graphs))])))
    enc = OHE(dtype=np.float32)
    return np.asarray(enc.fit_transform(targets.reshape(-1, 1)).todense())


def get_masks(N, train_size=0.6, val_size=0.2, random_seed=42):
    if not random_seed:
        seed = random.randint(0, 100)
    else:
        seed = random_seed

    # print('seed', seed)
    random.seed(seed)

    indices = list(range(N))
    random.shuffle(indices)

    train_mask = indices[:int(train_size * len(indices))]
    val_mask = indices[int(train_size * len(indices)):int((train_size + val_size) * len(indices))]
    train_val_mask = indices[:int((train_size + val_size) * len(indices))]
    test_mask = indices[int((train_size + val_size) * len(indices)):]

    return train_mask, val_mask, train_val_mask, test_mask

def feature_vector(X, y, train_mask, task='classification'):
    if task == 'classification':
        catboost_loss_function = 'MultiClass'
        catboost_object = CatBoostClassifier
    else:
        catboost_loss_function = 'RMSE'
        catboost_object = CatBoostRegressor

    model = catboost_object(iterations=100,
                                depth=6,
                                learning_rate=0.1,
                                loss_function=catboost_loss_function,
                                random_seed=0,
                                nan_mode='Min',
                                allow_const_label=True)
    
    X_train = X.iloc[train_mask]
    y_train = y.iloc[train_mask]

    model.fit(X_train, y_train, verbose=False)
    # prediction
    if task == 'classification':
        prediction = model.predict_proba(X)
    else:
        prediction = model.predict(X)
    # leaf index
    leaf_index = model.calc_leaf_indexes(X)
    return prediction, leaf_index

def construct_graph(nf):
    warnings.filterwarnings('ignore')   # ignore dgl warnings
    graph = dgl.DGLGraph()
    if isinstance(nf, np.ndarray):
        nf = pd.DataFrame(nf)
    nodes = list(nf.index)
    graph.add_nodes(len(nodes))

    simul = cosine_similarity(nf.values, nf.values)
    np.fill_diagonal(simul, 0)
    
    print("Use top5-KNN to construct graph")
    top5 = np.argpartition(simul, -5)[:, -5:]

    src_node = []
    dst_node = []
    for i in nodes:
        for j in top5[i]:
            src_node.append(i)
            dst_node.append(j)
    
    graph.add_edges(src_node, dst_node)
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)

    return graph

# corrupt labels to test the robustness
def corrupt_label(target_labels, train_mask, noise_rate=0.1):
    np.random.seed(42)
    noisy_y = target_labels.copy()
    n_samples = len(train_mask)
    n_noise = int(noise_rate * n_samples)
    noisy_idx = np.random.choice(train_mask, size=n_noise, replace=False)

    for idx in noisy_idx:
        # for binary classification task
        opposite_label = 1 - int(target_labels.loc[idx])
        noisy_y.loc[idx] = opposite_label

    return noisy_y

def get_noisy_y(input_folder, y, train_mask, noise_rate=0.1):
    # reload y in every seed
    y = pd.read_csv(f'{input_folder}/y.csv')

    print("robustness test with noise rate:", noise_rate)
    noisy_y = corrupt_label(y, train_mask, noise_rate=noise_rate)
    y = noisy_y.copy()

    return y

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)