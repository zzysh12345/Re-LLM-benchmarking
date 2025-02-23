import json
import os
import pickle
import sys

os.chdir('../')
print(os.getcwd())
project_root = os.getcwd()
sys.path.append(project_root)

from itertools import product
from LLM.utils import process_combination
import torch
from utils.utils import load_data
from tqdm import tqdm

# combination of parameters
params = {
    "dataset_name": ['cora', "pubmed", "arxiv", "product"],
    "mode": ["neighbors", "ego"],
    "zero_shot_CoT": [False],
    "hop": [1, 2],
    "include_label": [False]
}



# generate all possible combinations
combinations = list(product(
    params["dataset_name"],
    params["mode"],
    params["zero_shot_CoT"],
    params["hop"],
    params["include_label"]
))


# filter combinations based on rules
filtered_combinations = process_combination(combinations)
few_shot_key = {}
shot_count = 20

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

def run(combo):
    """
    execute tasks for different parameter combinations
    """
    try:
        dataset, mode, zero_shot_CoT, hop, include_label = combo
        suffix = '.jsonl'
        path = f'output/{dataset}/node_classification/test/'

        try:
            data, text = get_data_and_text(dataset)
        except Exception as e:
            print(f"Error in get_data_and_text for dataset {dataset}: {e}")
            return

        try:
            options = set(text['label'])
        except KeyError as e:
            print(f"Error accessing 'label' in text for dataset {dataset}: {e}")
            return

        if dataset not in few_shot_key.keys():
            few_shot_key[dataset] = {}

        for node_class in options:
            if node_class not in few_shot_key[dataset].keys():
                few_shot_key[dataset][node_class] = []

        few_shot_count = {}
        for node_class in options:
            if node_class not in few_shot_count.keys():
                few_shot_count[node_class] = 0

        # build path
        try:
            if mode == 'ego':
                if zero_shot_CoT:
                    path += f'ego_CoT'
                else:
                    path += f'ego'
            elif mode == 'neighbors':
                if include_label:
                    if zero_shot_CoT:
                        path += f'{hop}_hop_with_label_CoT'
                    else:
                        path += f'{hop}_hop_with_label'
                else:
                    if zero_shot_CoT:
                        path += f'{hop}_hop_without_label_CoT'
                    else:
                        path += f'{hop}_hop_without_label'
            else:
                path += 'pure_structure_'
                if include_label:
                    if zero_shot_CoT:
                        path += f'{hop}_hop_with_label_CoT'
                    else:
                        path += f'{hop}_hop_with_label'
                else:
                    if zero_shot_CoT:
                        path += f'{hop}_hop_without_label_CoT'
                    else:
                        path += f'{hop}_hop_without_label'
            path += suffix
        except Exception as e:
            print(f"Error constructing path for combo {combo}: {e}")
            return

        try:
            with open(path, 'r') as file:
                for line in tqdm(file, desc="Processing lines", unit="line"):
                    if min(few_shot_count.values()) >= shot_count:
                        return
                    try:
                        json_obj = json.loads(line.strip())
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in line: {line.strip()} - {e}")
                        continue

                    answer = json_obj['Answer']
                    if few_shot_count[answer] >= shot_count:
                        continue
                    few_shot_count[answer] += 1

                    context = json_obj['Context']
                    question = json_obj['Question']
                    json_data = {
                        "Context": context,
                        "Question": question,
                        "Answer": answer
                    }
                    few_shot_key[dataset][answer].append(json_data)

        except FileNotFoundError as e:
            print(f"Test file not found at path {path}: {e}")
            return

    except Exception as e:
        print(f"Unhandled error with combo {combo}: {e}")


if __name__ == "__main__":
    for item in tqdm(filtered_combinations, desc="Processing combinations"):
        try:

            dataset = item['dataset']
            mode = item['mode']
            zero_shot_CoT = item['zero_shot_CoT']
            hop = item['hop']
            include_label = item['include_label']

            # construct combo
            combo = (dataset, mode, zero_shot_CoT, hop, include_label)
            print(f"{combo}")

            run(combo)

        except Exception as e:
            print(f"Error processing {e}")


    cora = few_shot_key['cora']
    output_file = f'output/cora/few_shot/{shot_count}_shot.jsonl'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    result = [item for sublist in cora.values() for item in sublist]
    with open(output_file, "a") as f:
        for json_data in result:
            f.write(json.dumps(json_data) + "\n")


    pubmed = few_shot_key['pubmed']
    output_file = f'output/pubmed/few_shot/{shot_count}_shot.jsonl'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    result = [item for sublist in pubmed.values() for item in sublist]
    with open(output_file, "a") as f:
        for json_data in result:
            f.write(json.dumps(json_data) + "\n")


    product = few_shot_key['product']
    output_file = f'output/product/few_shot/{shot_count}_shot.jsonl'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    result = [item for sublist in product.values() for item in sublist]
    with open(output_file, "a") as f:
        for json_data in result:
            f.write(json.dumps(json_data) + "\n")



    arxiv = few_shot_key['arxiv']
    output_file = f'output/arxiv/few_shot/{shot_count}_shot.jsonl'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    result = [item for sublist in arxiv.values() for item in sublist]
    with open(output_file, "a") as f:
        for json_data in result:
            f.write(json.dumps(json_data) + "\n")





