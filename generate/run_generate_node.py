import os
import sys


os.chdir('../')
print(os.getcwd())
project_root = os.getcwd()
sys.path.append(project_root)
from concurrent.futures import ThreadPoolExecutor

from generate.node_classification_generate import run_node_classification_generate


from itertools import product

# 定义参数
params = {
    # "dataset_name": ['product'],
    "dataset_name": ['cora','pubmed','arxiv','product'],
    "is_train": [True, False],
    "mode": ["neighbors", "ego", "pure structure"],
    # "mode": ["pure structure"],
    "zero_shot_CoT": [True, False],
    "BAG": [True, False],
    "few_shot": [True, False],
    "hop": [1, 2],
    "include_label": [True, False]
}


combinations = list(product(
    params["dataset_name"],
    params["is_train"],
    params["mode"],
    params["zero_shot_CoT"],
    params["BAG"],
    params["few_shot"],
    params["hop"],
    params["include_label"]
))


filtered_combinations = []
for combo in combinations:
    dataset, is_train, mode, zero_shot_CoT, BAG, few_shot, hop, include_label = combo

    if is_train and zero_shot_CoT:
        continue

    if is_train and BAG:
        continue

    if is_train and few_shot:
        continue

    if (zero_shot_CoT + BAG + few_shot) == 1 or (not zero_shot_CoT and not BAG and not few_shot):
        pass
    else:
        continue

    if mode == "ego" and (include_label or hop != 1):
        continue

    filtered_combinations.append({
        "dataset": dataset,
        "is_train": is_train,
        "mode": mode,
        "zero_shot_CoT": zero_shot_CoT,
        "BAG": BAG,
        "few_shot": few_shot,
        "hop": hop,
        "include_label": include_label
    })
print(f"Number of combinations: {len(filtered_combinations)}", flush=True)

def process_combination(combo):
    dataset = combo['dataset']
    is_train = combo['is_train']
    mode = combo['mode']
    zero_shot_CoT = combo['zero_shot_CoT']
    BAG = combo['BAG']
    few_shot = combo['few_shot']
    hop = combo['hop']
    include_label = combo['include_label']

    if dataset == "cora":
        test_sample_size = 542
    else:
        test_sample_size = 1000

    arxiv_style = "subcategory"
    print(f"Running: {combo}", flush=True)

    run_node_classification_generate(
        dataset,
        arxiv_style,
        is_train,
        mode,
        zero_shot_CoT,
        hop,
        False,
        include_label,
        test_sample_size,
        BAG=BAG,
        few_shot=few_shot,
        include_options=True,
        save_dir='output')
# use ThreadPoolExecutor
if __name__ == "__main__":
    try:
        # with ThreadPoolExecutor(max_workers=8) as executor:
        #     executor.map(process_combination, filtered_combinations)
        combo_test = {
            "dataset": 'Amazon',
            "is_train": False,
            "mode": "neighbors",
            "zero_shot_CoT": True,
            "BAG": False,
            "few_shot": False,
            "hop": 1,
            "include_label": True
        }
        process_combination(combo_test)
    except Exception as e:
        print(e)