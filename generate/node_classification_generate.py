# from test_cora import sample_size
from utils.utils import load_data, get_and_save_message_for_node
import numpy as np
import os
import torch
import pickle
# os.chdir('../')
def get_data_and_text(dataset_name, train_perc=0.6, val_perc=0.2, test_perc=0.2, use_text=True, seed=42):
    # file path
    data_path = f"dataset/{dataset_name}_data.pt"
    text_path = f"dataset/{dataset_name}_text.pkl"

    # check if the files exist
    if os.path.exists(data_path) and os.path.exists(text_path):
        print("Load data and text files...")

        # load data and text
        data = torch.load(data_path)
        with open(text_path, "rb") as f:
            text = pickle.load(f)
    else:
        print("generate data and text files...")
        # regenerate data and text
        data, text = load_data(dataset_name, train_perc, val_perc, test_perc, use_text, seed)

        # save data and text
        torch.save(data, data_path)
        with open(text_path, "wb") as f:
            pickle.dump(text, f)
        print("data and text files saved!")

    return data, text
def run_node_classification_generate(dataset_name, arxiv_style, is_train, mode, zero_shot_CoT, hop, include_abs, include_label, test_sample_size, BAG=False, few_shot=False, include_options=True, save_dir='output'):
    dataset_name = dataset_name
    source = dataset_name
    arxiv_style=arxiv_style # "identifier", "natural language"

    # data, text = load_data(dataset_name, trian_perc=0.6, val_perc=0.2, test_perc=0.2, use_text=True, seed=42)
    data, text = get_data_and_text(dataset_name)
    print(data)

    options = set(text['label'])
    is_train = is_train
    if is_train:
        if dataset_name == 'arxiv':
            sample_size = 20000
        elif dataset_name == 'product':
            sample_size = 10000
        else:
            sample_size = data.train_mask.sum().item()
        sample_indices = np.random.choice(np.where(data.train_mask.numpy())[0], size=sample_size, replace=False).tolist()
    else:
        sample_size = test_sample_size
        sample_indices = np.random.choice(np.where(data.test_mask.numpy())[0], size=sample_size, replace=False).tolist()

    if dataset_name == "product":
        max_papers_1 = 30
        max_papers_2 = 10
    else:
        max_papers_1 = 20
        max_papers_2 = 5

    get_and_save_message_for_node(sample_indices, data, text, dataset_name, include_label=include_label, source=source, save_dir=save_dir, hop=hop, max_papers_1=max_papers_1, max_papers_2=max_papers_2, mode=mode, arxiv_style=arxiv_style, include_abs=include_abs, include_options=include_options, zero_shot_CoT=zero_shot_CoT, BAG=BAG, few_shot=few_shot, options=options, is_train=is_train)

