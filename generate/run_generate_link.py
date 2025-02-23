import os
import sys
os.chdir('../')
print(os.getcwd())
project_root = os.getcwd()
sys.path.append(project_root)

from concurrent.futures import ThreadPoolExecutor

from generate.link_prediction_generate import run_link_prediction_generate


from itertools import product


params = {
    "dataset_name": ['cora', 'pubmed', 'arxiv', 'product'],
    # "dataset_name": ['product'],
    "is_train": [True, False],
    "include_title": [True, False],
    "case": [0,1,2,3,4,5,6,7,8]
}


combinations = list(product(
    params["dataset_name"],
    params["is_train"],
    params["include_title"],
    params["case"]
))

filtered_combinations = []
for combo in combinations:
    dataset, is_train, include_title, case = combo

    if (not is_train) and case != 0 and case != 1:
        continue

    filtered_combinations.append({
        "dataset": dataset,
        "is_train": is_train,
        "include_title": include_title,
        "case": case
    })


def process_combination(combo):
    dataset = combo['dataset']
    is_train = combo['is_train']
    include_title = combo['include_title']
    case = combo['case']

    if dataset == "cora":
        test_sample_size = 542
    else:
        test_sample_size = 1000

    run_link_prediction_generate(
        dataset, is_train, test_sample_size, include_title, save_dir='output', case=case)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=32) as executor:
        executor.map(process_combination, filtered_combinations)