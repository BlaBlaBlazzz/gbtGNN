from __future__ import annotations

import warnings
from pathlib import Path

import yaml
from examples.toy_example.dummy_data_utils import (
    preprocess_dummy_data,
    toy_classification,
    toy_regression,
)
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import log_loss, roc_auc_score, root_mean_squared_error
from tabpfn import TabPFNClassifier, TabPFNRegressor

# Suppress specific FutureWarnings from sklearn
warnings.filterwarnings(
    "ignore",
    message=r".*BaseEstimator.*is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*'force_all_finite' was renamed to 'ensure_all_finite'.*",
    category=FutureWarning,
)


def run_modelling(
    *,
    time_limit: int,
    finetuning_config: dict,
    X,
    y,
    seed,
    task_type,
    device="cuda",
):
    """Run a dummy experiment to compare TabPFN and LGBM models.

    Parameter
    ---------
    time_limit: int
        Time limit for fine-tuning TabPFN model.
    X: pd.DataFrame
        Features
    y: pd.Series
        Target
    seed: int
        Random seed
    task_type: str
        Task type: binary, multiclass or regression
    device: str
        Device to run the experiment
    """
    print(f"Start Dummy Experiment for {task_type}")

    match task_type:
        case "binary":
            val_metric = "roc_auc"
            tabpfn_model = TabPFNClassifier
            lgbm_model = LGBMClassifier
            test_metric = lambda y_test, y_pred: roc_auc_score(y_test, y_pred[:, 1])
            predict_func = lambda model, X: model.predict_proba(X)
            lower_is_better = False
        case "multiclass":
            val_metric = "log_loss"
            tabpfn_model = TabPFNClassifier
            lgbm_model = LGBMClassifier
            # FIXME: internally, we always use predict_proba for retrieval!
            test_metric = log_loss
            predict_func = lambda model, X: model.predict_proba(X)
            lower_is_better = True
        case "regression":
            val_metric = "rmse"
            tabpfn_model = TabPFNRegressor
            lgbm_model = LGBMRegressor
            test_metric = root_mean_squared_error
            predict_func = lambda model, X: model.predict(X)
            lower_is_better = True
        case _:
            raise ValueError(f"Invalid task_type: {task_type}")

    (
        X_train,
        X_test,
        y_train,
        y_test,
        categorical_features,
        categorical_features_index,
    ) = preprocess_dummy_data(
        X=X,
        y=y,
        seed=seed,
        stratify=task_type != "regression",
    )

    from finetuning_scripts.finetune_tabpfn_main import fine_tune_tabpfn
    use_wandb = False

    if use_wandb:
        import wandb
        wandb.init(
            reinit=True,
            # set the wandb project where this run will be logged
            project="LA-TabPFN",
            entity="lennartpurucker",
            # track hyperparameters and run metadata
            config={"version": "0.0.1", "finetuning_config": finetuning_config},
        )
    save_path_to_fine_tuned_model = (
        Path(__file__).parent / f"finetuned_tabpfn_model_{task_type}.ckpt"
    )
    fine_tune_tabpfn(
        path_to_base_model="auto",
        save_path_to_fine_tuned_model=save_path_to_fine_tuned_model,
        # Finetuning HPs
        time_limit=time_limit,
        finetuning_config=finetuning_config,
        validation_metric=val_metric,
        # Input Data
        X_train=X_train,
        y_train=y_train,
        categorical_features_index=categorical_features_index,
        device=device,
        task_type=task_type,
        # Optional
        show_training_curve=True,
        logger_level=0,
        use_wandb=use_wandb,
    )

    # -- Run Models
    results = {}
    for model_name, model in [
        (
            "Finetuned-TabPFN",
            tabpfn_model(
                model_path=save_path_to_fine_tuned_model,
                random_state=seed,
                device=device,
                categorical_features_indices=categorical_features_index,
            ),
        ),
        ("LGBM", lgbm_model(seed=seed, verbosity=-1)),
        (
            "Default-TabPFN",
            tabpfn_model(
                random_state=seed,
                device=device,
                categorical_features_indices=categorical_features_index,
            ),
        ),
    ]:
        model.fit(X_train, y_train)
        results[model_name] = test_metric(y_test, predict_func(model, X_test))

    metric_txt = "↑" if not lower_is_better else "↓"
    report = f"""Experiment Results for [{task_type} ({metric_txt})]:
    - LGBM            : {results["LGBM"]:.4f}
    - Default TabPFN  : {results["Default-TabPFN"]:.4f}
    - Finetuned TabPFN: {results["Finetuned-TabPFN"]:.4f}
    """
    print(report)


if __name__ == "__main__":
    import torch

    with (Path(__file__).parent / "finetuning_hps.yaml").open("r") as file:
        finetuning_config = yaml.safe_load(file)

    x_y_task_type = []
    for n_classes, task_type in [
        (2, "binary"),
        (5, "multiclass"),
    ]:
        x, y = toy_classification(
            n_classes=n_classes,
            has_cat_features=True,
            cat_has_nan=True,
            num_has_nan=True,
            seed=42,
            cols=20,
        )
        x_y_task_type.append((x, y, task_type))

    x, y = toy_regression(
        has_cat_features=True,
        cat_has_nan=True,
        num_has_nan=True,
        seed=42,
        cols=30,
    )
    x_y_task_type.append((x, y, "regression"))

    for x, y, task_type in x_y_task_type:
        run_modelling(
            time_limit=int(60 * 2),
            finetuning_config=finetuning_config,
            X=x,
            y=y,
            seed=48,
            task_type=task_type,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
