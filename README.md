# gbtGNN

This repositary contains the implementation of the paper gbtGNN: Graph-Regularized Tabular Learning via Coupling Gradient Boosted Trees with Graph Neural Networks. gbtGNN is a hybrid end-to-end model of GBDT and GNN, which the concept is motivated by the ICLR 2021 paper [BGNN](https://openreview.net/pdf?id=ebS5NUfoMKL). We extend the graph-based model to tabular data and compare it with wide SOTA baselines.

## Installation

The implementation is based on python 3.9, to run all baselines other than TabPFN, please create the environment and install the required packages.

```bash
git clone https://github.com/BlaBlaBlazzz/gbtGNN.git
cd gbtGNN
pip install -r requirements.txt
```

To download `dgl`, please make sure you select the correct CUDA version for your setup.

```bash
# An example with dgl==1.0.1+cu117
pip install dgl==1.0.1+cu117 -f https://data.dgl.ai/wheels/cu117/repo.html
```

Since TabPFN require python version >= 3.10, please use other env to execute the `run_tabpfn.py` script.

```bash
pip install -r requirements_tabpfn.txt
```

## Datasets

We evaluate models on 49 small datasets used in ExcelFormer and Taptap benchmark. The datasets are available at [https://huggingface.co/datasets/jyansir/excelformer], or you can access the datasets via our `gen_datasets.py` script, which will download the datasets and generate the `X.csv`, `y.csv`, `cat_features.txt` and `masks.json` among traditional 622 data distribution and few shot scenarios.

```bash
cd data
python gen_datasets.py
```

## Usage

The main script is the `run.py`, the following is the format to run the file

```bash
python scripts/run.py [dataset] [model] (Optional)--task [task] --save_folder [location]
```

Here are a example to run gnn model on `Automobiles` dataset and store to the `./results` directory.

```bash
python scripts/run.py Automobiles gnn --task classification --save_folder ./results
```

After complete the process, the results will be saved to the target directory (Default: results/{dataset}/day_month/):
- `aggregated_results.json`: The aggregated results of each seeds, containing {`mean metric`, `std metric`, `Mean Time`, `std metric`}.
- `seed_results.json`: The results of each experiment and seed.
- `seed_folder`: The training info of each hyperparameter combination.


## Test your dataset and models

To test your model, place your model in the `./models` directory, and write a confiuration `.yaml` file of the model in `./config` directory. Keep the same pipeline we use in other baseline models.

To test your dataset, please make sure to have 
- `X.csv` : Features of the data
- `y.csv` : Target labels of the data
- `cat_features.txt` (Optional) : The names of categorical columns
- `masks.json` (Optional) : Indexes of train/valid/test data 