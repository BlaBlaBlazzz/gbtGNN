import json
import time
import random
import re
import os
import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset


def gen_masks(data, path):
    masks = {str(i):{"train":[], "val":[], "test":[]} for i in range(5)}
    scale_config = [int(0.6*len(data)), int(0.2*len(data)), int(0.2*len(data))]
    # print(scale_config)

    for key in list(masks.keys()):
        ids = list(data.index)
        random.shuffle(ids)

        masks[key]["train"] = ids[:scale_config[0]]
        masks[key]["val"] = ids[scale_config[0]:scale_config[0]+scale_config[1]]
        masks[key]["test"] = ids[scale_config[0]+scale_config[1]:]

    write_json(masks, path)

def sampling_masks(data, label, path, num):
    ids = list(data.index)
    masks = {str(i):{"train":[], "val":[], "test":[]} for i in range(5)}
    classes = int(max(label['class'])) + 1
    class_dict = {k:[] for k in range(classes)}
    [class_dict[int(label.iloc[i])].append(i) for i in ids]

    for key in list(masks.keys()):
        # shuffle ids
        [random.shuffle(class_dict[i]) for i in range(classes)]

        masks[key]["train"] = [item for i in range(classes) for item in class_dict[i][:num]]
        masks[key]["val"] = [item for i in range(classes) for item in class_dict[i][num:2*num]]
        masks[key]["test"] = [item for i in range(classes) for item in class_dict[i][2*num:]]

        # random shuffle again
        random.shuffle(masks[key]["train"])
        random.shuffle(masks[key]["val"])
        random.shuffle(masks[key]["test"])
    
    write_json(masks, path)
    return masks
        

def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


# processing huggingface small datasets
def data_process(dataset):
    data = {}
    for table_name, table, task in zip(dataset['dataset_name'], dataset['table'], dataset['task']):
        table = ast.literal_eval(table)
        if task == 'binclass':
            data[table_name] = {
                'X_num': None if not table['X_num'] else pd.DataFrame.from_dict(table['X_num']),
                'X_cat': None if not table['X_cat'] else pd.DataFrame.from_dict(table['X_cat']),
                'y': np.array(table['y']),
                'y_info': table['y_info'],
                'task': task,
            }
    return data

def key_edition(key):
    key = re.sub(r"\[(kaggle|openml|UCI)\]\s*", "", key)
    return key.replace(" ", "_").strip()



datasets = load_dataset('jyansir/excelformer')
train_data, val_data, test_data = datasets['train'].to_dict(), datasets['val'].to_dict(), datasets['test'].to_dict()

train_data = data_process(train_data)
val_data = data_process(val_data)
test_data = data_process(test_data)

for key in train_data.keys():
    print("key:", key)
    train = train_data[key]
    val = val_data[key]
    test = test_data[key]

    col_names = None
    combined_X_num = pd.concat([train['X_num'], val['X_num'], test['X_num']], axis=0, ignore_index=True)
    if train['X_cat'] is not None:
        combined_X_cat = pd.concat([train['X_cat'], val['X_cat'], test['X_cat']], axis=0, ignore_index=True)
        combined_X = pd.concat([combined_X_cat, combined_X_num], axis=1)
        col_names = list(train['X_cat'].columns)
    else:
        combined_X = combined_X_num

    combined_y = np.append(np.append(train['y'], val['y']), test['y'])
    combined_y = pd.DataFrame(combined_y, columns=['class'])

    path = key_edition(key)
    path = f'{path}'

    if os.path.exists(path):
        path = f"{path}_UCI"
        os.mkdir(path)
    else:
        os.mkdir(path)
    path = f"{path}/{path}"
    path10 = f"{path}_s10"
    path4 = f"{path}_s4"

    os.mkdir(path)
    os.mkdir(path10)
    os.mkdir(path4)

    # create masks.json
    print("---generating masks---")
    gen_masks(combined_X, f"{path}/masks.json")
    sampling_masks(combined_X, combined_y, path=f'{path10}/masks.json', num=10)  # s10
    sampling_masks(combined_X, combined_y, path=f'{path4}/masks.json', num=4)  # s4
    print("---masks.json finished---")

    path1 = [path, path10, path4]

    for path in path1:
        print("\npath:", path)
        start_time = time.time()

        combined_X.to_csv(f"{path}/X.csv", encoding='utf-8', index=False)
        combined_y.to_csv(f"{path}/y.csv", encoding='utf-8', index=False)

        if col_names:
            with open(f"{path}/cat_features.txt", 'w') as f:
                for name in col_names:
                    f.write(f"{name}\n")
