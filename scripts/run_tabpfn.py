from __future__ import annotations

import sys
import os
sys.path.append(f'{os.getcwd()}/scripts/finetune_tabpfn_v2')

import os
import json
import time
import datetime
import numpy as np
import torch
import fire
import pandas as pd
from pathlib import Path

from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from collections import defaultdict as ddict

from tabpfn_extensions.hpo import TunedTabPFNClassifier
from finetuning_scripts.finetune_tabpfn_main import fine_tune_tabpfn
from tabpfn import TabPFNClassifier
# from tabpfn_extensions.hpo import TunedTabPFNClassifier

import warnings
warnings.filterwarnings("ignore")

class tabPFN:
    def __init__(self, task='regression', lr=0.1, num_leaves=31, max_bin=255,
                 lambda_l1=0., lambda_l2=0., boosting='gbdt'):
        self.task = task

    def accuracy(self, preds, train_data):
        labels = train_data.get_label()
        # print(labels.shape)
        preds_classes = preds.reshape((preds.shape[0]//labels.shape[0], labels.shape[0])).argmax(0)
        return 'accuracy', accuracy_score(labels, preds_classes), True

    def r2(self, preds, train_data):
        labels = train_data.get_label()
        return 'r2', r2_score(labels, preds), True

    def save_metrics(self, metrics, fn):
        with open(fn, "w+") as f:
            for key, value in metrics.items():
                print(key, value, file=f)

    def train_val_test_split(self, X, y, train_mask, val_mask, test_mask):
        X_train, y_train = X.iloc[train_mask], y.iloc[train_mask]
        X_val, y_val = X.iloc[val_mask], y.iloc[val_mask]
        X_test, y_test = X.iloc[test_mask], y.iloc[test_mask]
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def evaluate(self, X_train, y_train, X_val, y_val, X_test, y_test, metrics):
        # train evaluation
        preds = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, preds)
        # val evaluation
        preds = self.model.predict(X_val)
        val_acc = accuracy_score(y_val, preds)
        # test evaluation
        preds = self.model.predict(X_test)
        test_acc = accuracy_score(y_test, preds)

        metrics.append([train_acc, val_acc, test_acc])

        return metrics
    
    def fintune_tabpfn(self, X_train, y_train):
        save_path_to_fine_tuned_model = "./fine_tuned_model.ckpt"
        fine_tune_tabpfn(
            path_to_base_model="auto",
            save_path_to_fine_tuned_model=save_path_to_fine_tuned_model,
            # Finetuning HPs
            time_limit=60,
            finetuning_config={"learning_rate": 0.00001, "batch_size": 20},
            validation_metric="log_loss",
            # Input Data
            X_train=X_train,
            y_train=y_train.iloc[:, 0],
            categorical_features_index=None,
            device="cuda",  # use "cpu" if you don't have a GPU
            task_type="binary",
            # Optional
            show_training_curve=True,  # Shows a final report after finetuning.
            logger_level=0,  # Shows all logs, higher values shows less
            use_wandb=False,  # Init wandb yourself, and set to True
        )

        return save_path_to_fine_tuned_model

        


    def fit(self,
            X, y, train_mask, val_mask, test_mask,
            cat_features=None, num_epochs=1000, patience=200,
            loss_fn="", metric_name='loss'):

        metrics = []
        X_train, y_train, X_val, y_val, X_test, y_test = \
            self.train_val_test_split(X, y, train_mask, val_mask, test_mask)

        start = time.time()

        # save_path_to_fine_tuned_model = self.fintune_tabpfn(X_train, y_train)
        # self.model = TabPFNClassifier(model_path=save_path_to_fine_tuned_model)

        self.model = TunedTabPFNClassifier(
            n_trials=15,                    # Number of hyperparameter configurations to try
            metric='accuracy',              # Metric to optimize
            random_state=42                 # For reproducibility
        )
        
        self.model.fit(X_train, y_train.iloc[:, 0])

        self.evaluate(X_train, y_train, X_val, y_val, X_test, y_test, metrics)

        finish = time.time()
        print('Finished training. Total time: {:.2f}'.format(finish - start))
        return metrics

    def predict(self, X_test, y_test):
        pred = self.model.predict(X_test)

        metrics = {}
        metrics['rmse'] = mean_squared_error(pred, y_test) ** .5

        return metrics


class RunModel:
    def __init__(self):
        super(RunModel, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def read_input(self, input_folder):
        self.X = pd.read_csv(f'{input_folder}/X.csv')
        self.y = pd.read_csv(f'{input_folder}/y.csv')

        if os.path.exists(f'{input_folder}/masks.json'):
            with open(f'{input_folder}/masks.json') as f:
                self.masks = json.load(f)
        
    

    def get_input(self, dataset_dir, dataset: str):
        input_folder = dataset_dir / f'{dataset}_s4'  # few shot learning
        print(input_folder)

        if self.save_folder is None:
            self.save_folder = f'results_v2/{dataset}/{datetime.datetime.now().strftime("%d_%m")}'

        self.read_input(input_folder)
        print('Save to folder:', self.save_folder)
    
    def create_save_folder(self, seed):
        self.seed_folder = f'{self.save_folder}/{seed}'
        os.makedirs(self.seed_folder, exist_ok=True)

    def split_masks(self, seed):
        self.train_mask, self.val_mask, self.test_mask = self.masks[seed]['train'], \
                                                         self.masks[seed]['val'], self.masks[seed]['test']

    def save_results(self, seed, metrics):
        self.seed_results[seed] = metrics
        print("seed result", self.seed_results[seed])
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
        algos = ['tabpfn']
        model_best_score = ddict(list)
        model_best_time = ddict(list)

        results = self.seed_results
        for seed in results:
            model_name = 'tabpfn'
            model_results_for_seed = ddict(list)
            for output in results[seed]:
                if self.task == 'regression': # rmse metric
                    val_metric, test_metric, time = output[0][1], output[0][2], output[2]
                else: # accuracy metric
                    val_metric, test_metric = output[1], output[2]
                model_results_for_seed[model_name].append((val_metric, test_metric))

            for model_name, model_results in model_results_for_seed.items():
                print(model_results)
                if self.task == 'regression':
                    best_result = min(model_results) # rmse
                else:
                    best_result = max(model_results) # accuracy
                model_best_score[model_name].append(best_result[1])

        aggregated = dict()
        for model, scores in model_best_score.items():
            aggregated[model] = (np.nanmean(scores), np.nanstd(scores),
                                 0, 0)
        print(aggregated)
        return aggregated

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

        dataset_dir = Path(dataset_dir) if dataset_dir else Path(__file__).parent.parent / 'datasets'
        config_dir = Path(config_dir) if config_dir else Path(__file__).parent.parent / 'configs' / 'model'
        print(dataset_dir, config_dir)

        self.task = task
        self.save_folder = save_folder
        self.get_input(dataset_dir, dataset)

        self.seed_results = dict()
        runs_custom = []
        for ix, seed in enumerate(self.masks):
            print(f'{dataset} Seed {seed}')
            self.seed = seed
            self.create_save_folder(seed)
            self.split_masks(seed)

            self.store_results = dict()
            times = []
            for _ in range(self.repeat_exp):
                start = time.time()
                inputs = {'X': self.X, 'y': self.y, 'train_mask': self.train_mask,
                          'val_mask': self.val_mask, 'test_mask': self.test_mask}
                
                model = tabPFN(self.task)
                metrics = model.fit(**inputs)

                finish = time.time()
                times.append(finish - start)

                print(runs_custom)     

            self.save_results(seed, metrics)
            if ix+1 >= max_seeds:
                break

        print(f'Finished {dataset}: {time.time() - start2run} sec.')

if __name__ == '__main__':
    fire.Fire(RunModel().run) 
