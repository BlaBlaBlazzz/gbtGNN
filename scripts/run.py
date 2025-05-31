import sys
import os
curPath = os.path.dirname(os.path.dirname(__file__))
sys.path.append(curPath)

from models.GBDT import GBDTCatBoost, GBDTLGBM, GBDTXGBoost
from models.RandomForest import RandomForest
from models.MLP import MLP
from models.GNN import GNN
from models.BGNN import BGNN
from models.ExcelFormer import ExcelFormer
from models.trompt import trompt
from models.tabnet import tabnet
from models.tabtransformer import tabtransformer
from models.fttransformer import fttransformer
from models.gbtGNN import gbtGNN
from models.t2gFormer import t2gFormer
from scripts.utils import *
from models.Base import BaseModel

import os
import json
import time
import datetime
from pathlib import Path
from collections import defaultdict as ddict

import pandas as pd
import networkx as nx
import torch
import dgl
import random
import warnings
import numpy as np
import fire
from omegaconf import OmegaConf
from sklearn.model_selection import ParameterGrid
import xgboost as xgb

class RunModel:
    def __init__(self):
        super(RunModel, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def read_input(self, input_folder):
        self.X = pd.read_csv(f'{input_folder}/X.csv')
        self.y = pd.read_csv(f'{input_folder}/y.csv')            

        categorical_columns = []
        if os.path.exists(f'{input_folder}/cat_features.txt'):
            with open(f'{input_folder}/cat_features.txt') as f:
                for line in f:
                    if line.strip():
                        categorical_columns.append(line.strip())

        self.cat_features = None
        if categorical_columns:
            columns = self.X.columns
            self.cat_features = np.where(columns.isin(categorical_columns))[0]

            for col in list(columns[self.cat_features]):
                self.X[col] = self.X[col].astype(str)


        if os.path.exists(f'{input_folder}/masks.json'):
            with open(f'{input_folder}/masks.json') as f:
                self.masks = json.load(f)
        else:
            print('Creating and saving train/val/test masks')
            idx = list(range(self.y.shape[0]))
            self.masks = dict()
            for i in range(self.max_seeds):
                random.shuffle(idx)
                r1, r2, r3 = idx[:int(.6*len(idx))], idx[int(.6*len(idx)):int(.8*len(idx))], idx[int(.8*len(idx)):]
                self.masks[str(i)] = {"train": r1, "val": r2, "test": r3}

            with open(f'{input_folder}/masks.json', 'w+') as f:
                json.dump(self.masks, f, cls=NpEncoder)


    def get_input(self, dataset_dir, dataset: str):
        input_folder = dataset_dir / dataset / dataset
        print(input_folder)

        if self.save_folder is None:
            self.save_folder = f'results/{dataset}/{datetime.datetime.now().strftime("%d_%m")}'

        self.read_input(input_folder)
        print('Save to folder:', self.save_folder)


    def run_one_model(self, config_fn, model_name):
        self.config = OmegaConf.load(config_fn)
        grid = ParameterGrid(dict(self.config.hp))
        for ps in grid:
            param_string = ''.join([f'-{key}{ps[key]}' for key in ps])
            exp_name = f'{model_name}{param_string}'
            print(f'\nSeed {self.seed} RUNNING:{exp_name}')

            runs = []
            runs_custom = []
            times = []
            for _ in range(self.repeat_exp):
                start = time.time()
                model = self.define_model(model_name, ps)
                
                inputs = {'X': self.X, 'y': self.y, 'train_mask': self.train_mask,
                          'val_mask': self.val_mask, 'test_mask': self.test_mask, 'cat_features': self.cat_features}
                # graph
                if model_name in ['gnn', 'resgnn', 'bgnn']:
                    inputs['networkx_graph'] = self.graph
                elif model_name in ['gbtGNN']:
                    inputs['graph'] = self.graph
                    inputs['graph_pred'] = self.graph_pred
                    inputs['graph_leaf'] = self.graph_leaf
                
                metrics = model.fit(num_epochs=self.config.num_epochs, patience=self.config.patience,
                           loss_fn=f"{self.seed_folder}/{exp_name}.txt",
                           metric_name='loss' if self.task == 'regression' else 'accuracy', **inputs)
                finish = time.time()

                best_loss = min(metrics['loss'], key=lambda x: x[1])
                best_custom = max(metrics['r2' if self.task == 'regression' else 'accuracy'], key=lambda x: x[1])
                runs.append(best_loss)
                runs_custom.append(best_custom)
                times.append(finish - start)
            self.store_results[exp_name] = (list(map(np.mean, zip(*runs))),
                                       list(map(np.mean, zip(*runs_custom))),
                                       np.mean(times),
                                       )
    
    def define_model(self, model_name, ps):
        if model_name == 'catboost':
            return GBDTCatBoost(self.task, **ps)
        elif model_name == 'lightgbm':
            return GBDTLGBM(self.task, **ps)
        elif model_name == 'xgboost':
            return GBDTXGBoost(self.task, **ps)
        elif model_name == 'mlp':
            return MLP(self.task, **ps)
        elif model_name == 'gnn':
            return GNN(self.task, **ps)
        elif model_name == 'emb-GBDT':
            model = GNN(self.task)
            x = model.pandas_to_torch(self.X.astype("float"))[0]
            node_features = model.init_node_features(x, False)
            # node embedding
            model.fit(self.X, self.y, self.train_mask, self.val_mask, self.test_mask,
                      cat_features=self.cat_features, networkx_graph=self.graph, 
                      num_epochs=1000, patience=100,
                      metric_name='loss' if self.task == 'regression' else 'accuracy')
            node_embedding = model.model(self.graph, node_features).detach().cpu().numpy()
            return GBDTCatBoost(self.task, **ps, gnn_embedding=node_embedding)
        
        elif model_name == 'resgnn':
            gbdt = GBDTCatBoost(self.task)
            gbdt.fit(self.X, self.y, self.train_mask, self.val_mask, self.test_mask,
                     cat_features=self.cat_features,
                     num_epochs=1000, patience=100,
                     plot=False, verbose=False, loss_fn=None,
                     metric_name='loss' if self.task == 'regression' else 'accuracy')
            return GNN(task=self.task, gbdt_predictions=gbdt.model.predict(self.X), **ps)
        
        elif model_name == 'bgnn':
            return BGNN(self.task, **ps)
        elif model_name == 'ExcelFormer':
            return ExcelFormer(self.task, **ps)
        elif model_name == 't2gFormer':
            return t2gFormer(self.task, **ps)
        elif model_name == 'gbtGNN':
            return gbtGNN(self.task, **ps)
        else:
            module = globals()[model_name]
            return module(self.task, **ps)

    def create_save_folder(self, seed):
        self.seed_folder = f'{self.save_folder}/{seed}'
        os.makedirs(self.seed_folder, exist_ok=True)

    def split_masks(self, seed):
        self.train_mask, self.val_mask, self.test_mask = self.masks[seed]['train'], \
                                                         self.masks[seed]['val'], self.masks[seed]['test']

    def save_results(self, seed):
        self.seed_results[seed] = self.store_results
        print(self.seed_results)
        with open(f'{self.save_folder}/seed_results.json', 'w+') as f:
            json.dump(self.seed_results, f)

        self.aggregated = self.aggregate_results()
        save_path = f'{self.save_folder}/aggregated_results.json'

        if os.path.exists(save_path):
            # update model results
            with open(save_path, 'r') as f:
                metrics = json.load(f)
            metrics.update(self.aggregated)

            with open(save_path, 'w+') as f:
                json.dump(metrics, f)
        else:
            with open(save_path, 'w+') as f:
                json.dump(self.aggregated, f)

    def get_model_name(self, exp_name: str, algos: list):
        # get name of the model (for gnn-like models (eg. gat))
        if 'name' in exp_name:
            model_name = '-' + [param[4:] for param in exp_name.split('-') if param.startswith('name')][0]
        elif 'mixup' in exp_name:
            model_name = '-Mixup' + [param[5:] for param in exp_name.split('-') if param.startswith('mixup')][0]
        else:
            model_name = ''

        # get a model used a MLP (eg. MLP-GNN)
        if 'gnn' in exp_name and 'mlpTrue' in exp_name:
            model_name += '-MLP'

        # algo corresponds to type of the model (eg. gnn, resgnn, bgnn)
        for algo in algos:
            if algo in exp_name.split("-"):
                return  algo + model_name
        return 'unknown'

    def aggregate_results(self):
        algos = ['catboost', 'lightgbm', 'xgboost', 'RandomForest', 'emb-GBDT', 'tabpfn',
                 'ExcelFormer', 'trompt', 'fttransformer', 'tabnet', 'tabtransformer', 
                 't2gFormer', 'gnn', 'resgnn', 'bgnn', 'gbtGNN']
        model_best_score = ddict(list)
        model_best_time = ddict(list)

        results = self.seed_results
        for seed in results:
            model_results_for_seed = ddict(list)
            for name, output in results[seed].items():
                model_name = self.get_model_name(name, algos=algos)
                if self.task == 'regression': # rmse metric
                    val_metric, test_metric, time = output[0][1], output[0][2], output[2]
                else: # accuracy metric
                    val_metric, test_metric, time = output[1][1], output[1][2], output[2]
                model_results_for_seed[model_name].append((val_metric, test_metric, time))

            for model_name, model_results in model_results_for_seed.items():
                if self.task == 'regression':
                    best_result = min(model_results) # rmse
                else:
                    best_result = max(model_results) # accuracy
                model_best_score[model_name].append(best_result[1])
                model_best_time[model_name].append(best_result[2])

        aggregated = dict()
        for model, scores in model_best_score.items():
            aggregated[model] = (np.nanmean(scores), np.nanstd(scores),
                                 np.nanmean(model_best_time[model]), np.nanstd(model_best_time[model]))
        return aggregated
    
    def get_graph(self, input_folder):
        if os.path.exists(f'{input_folder}/graph.graphml'):
            networkx_graph = nx.read_graphml(f'{input_folder}/graph.graphml')
            networkx_graph = nx.relabel_nodes(networkx_graph, {str(i): i for i in range(len(networkx_graph))})
            self.graph = dgl.from_networkx(networkx_graph)
            self.graph = dgl.remove_self_loop(self.graph)
            self.graph = dgl.add_self_loop(self.graph)
            self.graph = self.graph.to(self.device)

        elif os.path.exists(f'{input_folder}/cat_features.txt'):
            # normalize and encode categorical feature
            encoded_X = BaseModel().encode_cat_features(self.X, self.y, self.cat_features, self.train_mask, 
                                                        self.val_mask, self.test_mask)
            encoded_X = BaseModel().normalize_features(encoded_X, self.train_mask, self.val_mask, self.test_mask)
            self.graph = construct_graph(encoded_X)
            self.graph = self.graph.to(self.device)
        else:
            self.graph = construct_graph(self.X)
        
        pred_matrix, leaf_index = feature_vector(self.X, self.y, self.train_mask, self.task)
        self.graph_pred = construct_graph(pred_matrix)
        self.graph_leaf = construct_graph(leaf_index)        

        self.graph = self.graph.to(self.device)
        self.graph_pred = self.graph_pred.to(self.device)
        self.graph_leaf = self.graph_leaf.to(self.device)

    def run(self, dataset: str, *args,
            save_folder: str = None,
            task: str = 'regression',
            repeat_exp: int = 1,
            max_seeds: int = 5,
            dataset_dir: str = None,
            config_dir: str = None
            ):
        start2run = time.time()
        self.repeat_exp = repeat_exp
        self.max_seeds = max_seeds
        print(dataset, args, task, repeat_exp, max_seeds, dataset_dir, config_dir)

        dataset_dir = Path(dataset_dir) if dataset_dir else Path(__file__).parent.parent / 'data'
        config_dir = Path(config_dir) if config_dir else Path(__file__).parent.parent / 'configs' / 'model'
        print(dataset_dir, config_dir)

        self.task = task
        self.save_folder = save_folder
        self.get_input(dataset_dir, dataset)

        noise_rate = None
        self.seed_results = dict()
        for ix, seed in enumerate(self.masks):
            print(f'{dataset} Seed {seed}')
            self.seed = seed
            
            self.create_save_folder(seed)
            self.split_masks(seed)

            # robustness test
            if noise_rate != None:
                input_folder = dataset_dir / f'{dataset}'
                self.y = get_noisy_y(input_folder, self.y, self.train_mask, noise_rate)

            # construct graph
            self.get_graph(dataset_dir/dataset)
            
            self.store_results = dict()
            for arg in args:
                if arg == 'all':
                    self.run_one_model(config_fn=config_dir / 'bgnn.yaml', model_name="bgnn")
                    self.run_one_model(config_fn=config_dir / 'resgnn.yaml', model_name="resgnn")
                    self.run_one_model(config_fn=config_dir / 'emb-GBDT.yaml', model_name="emb-GBDT")
                    self.run_one_model(config_fn=config_dir / 'catboost.yaml', model_name="catboost")
                    self.run_one_model(config_fn=config_dir / 'ExcelFormer.yaml', model_name='ExcelFormer')
                    self.run_one_model(config_fn=config_dir / 'trompt.yaml', model_name='trompt')
                    self.run_one_model(config_fn=config_dir / 'tabnet.yaml', model_name='tabnet')
                    self.run_one_model(config_fn=config_dir / 'tabtransformer.yaml', model_name='tabtransformer')
                    self.run_one_model(config_fn=config_dir / 'fttransformer.yaml', model_name='fttransformer')
                    self.run_one_model(config_fn=config_dir / 'gnn.yaml', model_name="gnn")
                    self.run_one_model(config_fn=config_dir / 'xgboost.yaml', model_name="xgboost")
                    self.run_one_model(config_fn=config_dir / 'lightgbm.yaml', model_name="lightgbm")
                    self.run_one_model(config_fn=config_dir / 'RandomForest.yaml', model_name="RandomForest")
                    self.run_one_model(config_fn=config_dir / 'gbtGNN.yaml', model_name="gbtGNN")
                elif arg == 'catboost':
                    self.run_one_model(config_fn=config_dir / 'catboost.yaml', model_name="catboost")
                elif arg == 'xgboost':
                    self.run_one_model(config_fn=config_dir / 'xgboost.yaml', model_name="xgboost")
                elif arg == 'lightgbm':
                    self.run_one_model(config_fn=config_dir / 'lightgbm.yaml', model_name="lightgbm")
                elif arg == 'RandomForest':
                    self.run_one_model(config_fn=config_dir / 'RandomForest.yaml', model_name="RandomForest")
                elif arg == 'emb-GBDT':
                    self.run_one_model(config_fn=config_dir / 'emb-GBDT.yaml', model_name="emb-GBDT")
                elif arg == 'gnn':
                    self.run_one_model(config_fn=config_dir / 'gnn.yaml', model_name="gnn")
                elif arg == 'resgnn':
                    self.run_one_model(config_fn=config_dir / 'resgnn.yaml', model_name="resgnn")
                elif arg == 'bgnn':
                    self.run_one_model(config_fn=config_dir / 'bgnn.yaml', model_name="bgnn")
                elif arg == 't2gFormer':
                    self.run_one_model(config_fn=config_dir / 't2gFormer.yaml', model_name='t2gFormer')
                elif arg == 'gbtGNN':
                    self.run_one_model(config_fn=config_dir / 'gbtGNN.yaml', model_name="gbtGNN")
                else:
                    config_fn = config_dir / f'{arg}.yaml'
                    self.run_one_model(config_fn=config_fn, model_name=arg)
                
            self.save_results(seed)
            if ix+1 >= max_seeds:
                break

        print(f'Finished {dataset}: {time.time() - start2run} sec.')

if __name__ == '__main__':
    fire.Fire(RunModel().run) 