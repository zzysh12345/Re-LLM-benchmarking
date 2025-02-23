import os
import sys
from random import random

os.chdir('../../')
print(os.getcwd())
project_root = os.getcwd()
sys.path.append(project_root)
import json
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from tqdm import tqdm

from LLM.utils import process_combination, split_input_text, get_matched_option
from generate.node_classification_generate import get_data_and_text

qwen_api_key = "your_key"
model_name = 'qwen-plus'
def get_completion_from_messages(messages,
                                 model=model_name,
                                 temperature=0,
                                 max_tokens=1000):
    """
    Get completion from the OpenAI API based on the given messages.

    Parameters:
        messages (list): Messages to be sent to the OpenAI API.
        model (str, optional): The name of the model to be used. Default is "gpt-3.5-turbo".
        temperature (float, optional): Sampling temperature. Default is 0.
        max_tokens (int, optional): Maximum number of tokens for the response. Default is 500.

    Returns:
        str: The content of the completion message.
    """
    from openai import OpenAI
    # qwen_api_key = "sk-94bb5a57c62d47ea8afb2e4cd60edf8c"

    try:
        client = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key=qwen_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        completion = client.chat.completions.create(
            model=model,  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        # print(completion.choices[0].message.content)
        return completion.choices[0].message.content
        # print(completion)
    except Exception as e:
        print(f"错误信息：{e}")
        print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")

# params = {
#     # "dataset_name": ['cora', 'pubmed', 'arxiv', 'product'],
#     "dataset_name": ['cora','pubmed'],
#     "mode": ["neighbors", "ego"],
#     "zero_shot_CoT": [True],
#     "hop": [1, 2],
#     "include_label": [False, True]
# }

params = {
    # "dataset_name": ['pubmed'],
    "dataset_name": ['cora', 'pubmed', 'arxiv', 'product'],
    "is_train": [False],
    "mode": ["neighbors", "ego"],
    # "mode": ["pure_structure"],
    "zero_shot_CoT": [True, False],
    "BAG": [True, False],
    "few_shot": [True, False],
    "hop": [1, 2],
    "include_label": [False]
}

# 生成所有可能的组合
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

# 根据规则过滤
filtered_combinations = process_combination(combinations)
print(f"Filtered combinations: {filtered_combinations}")
# combo = filtered_combinations[0]
def run(combo):
    try:
        dataset = combo['dataset']
        mode = combo['mode']
        zero_shot_CoT = combo['zero_shot_CoT']
        BAG = combo['BAG']
        few_shot = combo['few_shot']
        hop = combo['hop']
        include_label = combo['include_label']
        suffix = '.jsonl'
        path = f'output/{dataset}/node_classification/test/'

        data, text = get_data_and_text(dataset)
        print(data)
        options = set(text['label'])

        try:
            if mode == 'ego':
                if zero_shot_CoT:
                    path += f'ego_CoT'
                elif BAG:
                    path += f'ego_BAG'
                elif few_shot:
                    path += f'ego_few_shot'
                else:
                    path += f'ego'
            elif mode == 'neighbors':
                if include_label:
                    path += f'{hop}_hop_with_label'
                else:
                    path += f'{hop}_hop_without_label'
            else:
                path += 'pure_structure_'
                if include_label:
                    path += f'{hop}_hop_with_label'
                else:
                    path += f'{hop}_hop_without_label'
            path += suffix
        except Exception as e:
            print(f"Error constructing path for combo {combo}: {e}")
            return

        # load the test file
        count = 0
        right = 0
        # path += suffix
        print(path)

        # 打开文件并加载所有行
        with open(path, 'r') as file:
            for line in tqdm(file, desc="Processing lines", unit="line"):
                if count > 500:
                    break
                try:
                    json_obj = json.loads(line.strip())
                except json.JSONDecodeError:
                    print("Error decoding JSON in line:", line)
                context = json_obj['Context']
                question = json_obj['Question']
                answer = json_obj['Answer']
                system_content, user_content = split_input_text(context)
                message = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content + '\n' + question}
                ]
                reply = get_completion_from_messages(message)
                prediction_answer = get_matched_option(reply, options)
                if prediction_answer is not None:
                    right += int(prediction_answer.lower() == answer.lower())
                count += 1

        try:
            accuracy = right / count if count > 0 else 0
            print(f"accuracy: {right}/{count} ({accuracy:.2%})")
        except ZeroDivisionError as e:
            print(f"Error calculating accuracy: {e}")
            accuracy = 0
        # 将结果写入log文件
        try:
            with open(f'results/{model_name}_log.txt', 'a') as f:
                f.write(f"{path} accuracy: {right}/{count} ({accuracy:.2%})\n")
        except Exception as e:
            print(f"Error writing to log file: {e}")



    except Exception as e:
        print(f"Error with combo {combo}: {e}", flush=True)
# 使用 ThreadPoolExecutor
if __name__ == "__main__":
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:  # 设置最大线程数为 4
            executor.map(run, filtered_combinations)
    except Exception as e:
        print(e)
    # process_combination(combo)
