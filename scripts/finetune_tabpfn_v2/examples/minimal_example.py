from __future__ import annotations

import sys
import os
sys.path.append(f'{os.getcwd()}')

from finetuning_scripts.finetune_tabpfn_main import fine_tune_tabpfn
from sklearn.datasets import load_iris
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier

# Load data
X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
)
print(y_train)

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
    y_train=y_train,
    categorical_features_index=None,
    device="cuda",  # use "cpu" if you don't have a GPU
    task_type="multiclass",
    # Optional
    show_training_curve=True,  # Shows a final report after finetuning.
    logger_level=0,  # Shows all logs, higher values shows less
    use_wandb=False,  # Init wandb yourself, and set to True
)

# Evaluate on Test Data
clf = TabPFNClassifier(model_path=save_path_to_fine_tuned_model).fit(X_train, y_train)
print("Log Loss (Finetuned):", log_loss(y_test, clf.predict_proba(X_test)))

# Compare to the default model
clf = TabPFNClassifier().fit(X_train, y_train)
print("Log Loss (Default):", log_loss(y_test, clf.predict_proba(X_test)))
