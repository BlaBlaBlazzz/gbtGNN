# gbtGNN

This repositary contains the implementation of the paper gbtGNN: Graph-Regularized Tabular Learning via Coupling Gradient Boosted Trees with Graph Neural Networks. gbtGNN is a hybrid end-to-end model of GBDT and GNN, which the concept is motivated by the ICLR 2021 paper [BGNN](https://openreview.net/pdf?id=ebS5NUfoMKL). We extend the graph-based model to tabular data and evaluate our method against a wide range of strong SOTA baselines.

## Installation

The implementation is based on python 3.9, to run all baselines except for TabPFN, please create a virtual environment and install the required packages:

```bash
git clone https://github.com/BlaBlaBlazzz/gbtGNN.git
cd gbtGNN
pip install -r requirements.txt
```

To install `dgl`, please ensure you select the correct CUDA version for your setup. For tested. For example, to install DGL with CUDA 11.7(Our version):

```bash
pip install dgl==1.0.1+cu117 -f https://data.dgl.ai/wheels/cu117/repo.html
```

Note: TabPFN require python version ≥ 3.10, please use a separate environment and install dependencies via:

```bash
pip install -r requirements_tabpfn.txt
```

## Datasets

We evaluate our models on 49 small-scale datasets used in ExcelFormer and Taptap benchmark. You can dowload them via:
- HuggingFace: https://huggingface.co/datasets/jyansir/excelformer
- Automatically using provided script:

```bash
cd data
python gen_datasets.py
```

This will generate:
- `X.csv`
- `y.csv`
- `cat_features.txt`
- `masks.json`

Under traditional 622 data distribution and few shot scenarios.


## Usage

The main script is the `run.py`. The general usage format is:

```bash
python scripts/run.py [dataset] [model] (Optional)--task [task] --save_folder [location]
```

Here is an example to run gnn model on `Automobiles` dataset and store to the `./results` directory.

```bash
python scripts/run.py Automobiles gnn --task classification --save_folder ./results
```

After execution, the results will be saved in the target directory (Default: `results/{dataset}/day_month/`):
- `aggregated_results.json`: Aggregated results accross seeds, containing `mean metric`, `std metric`, `Mean Time`, `std metric` in order.
- `seed_results.json`: The results of each experiment and seed.
- `seed_folder`: Logs and details of each hyperparameter combination


## Test your dataset and models

To evaluate your **model**:

1. place your model under `./models` directory, import it in `scripts/run.py` file.
2. Add a configuration `.yaml` file for your model in `./config` directory.
3. Follow the same pipeline as used in other baseline models.

To evaluate your **dataset**, ensure the following files are prepared:

- `X.csv` : Features of the data
- `y.csv` : Target labels of the data
- `cat_features.txt` (Optional) : The names of categorical columns
- `masks.json` (Optional) : Indexes of train/valid/test data 


## Reference

This repository includes the following external works as baselines:

- **[BGNN](https://github.com/nd7141/bgnn/tree/master)**. we refer to its pipeline structure and model designs.
  Copyright (c) 2024 russellsparadox, Licensed under MIT License.

- **[T2GFormer](https://github.com/jyansir/t2g-former)**. Included under `models/t2gformer` directory.
  Copyright(c) 2023 LionSenSei. Licensed under MIT License.

- **[Tabpfn_finetune](https://github.com/LennartPurucker/finetune_tabpfn_v2)**, included under `scripts/finetune_tabpfn_v2` directory.
  Copyright (c) 2025, Lennart Purucker. Licensed under BSD 3-Clause License.

- **[ExcelFormer Dataset](https://huggingface.co/datasets/jyansir/excelformer)**. Datasets used in our experiments are partially derived from this source.

- **[Pytorch-frame](https://github.com/pyg-team/pytorch-frame)**. Used for implementing DNN baselines.
  Copyright (c) 2023 PyG Team <team@pyg.org>. Licensed under MIT License.