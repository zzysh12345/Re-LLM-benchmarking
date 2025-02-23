import os
import sys
os.chdir('../../')
print(os.getcwd())
project_root = os.getcwd()
sys.path.append(project_root)
import json
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from tqdm import tqdm

from LLM.utils import process_combination, split_input_text, get_matched_option, split_input_text_link
from generate.node_classification_generate import get_data_and_text

qwen_api_key = "your_key"
model_name = 'qwen_plus'
def get_completion_from_messages(messages,
                                 model=model_name,
                                 temperature=0,
                                 max_tokens=5000):
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


def process_combination(combo):
    try:
        dataset = combo['dataset']
        case = combo['case']
        # mode = combo['mode']
        # zero_shot_CoT = combo['zero_shot_CoT']
        # hop = combo['hop']
        # include_label = combo['include_label']
        suffix = '.jsonl'
        path = f'output/{dataset}/link_prediction/test/case{case}'

        options = {'yes', 'no'}

        # load the test file
        count = 0
        right = 0
        path += suffix
        with open(path, 'r') as file:
            for line in tqdm(file, desc="Processing lines", unit="line"):
                if count > 1000:
                    break
                try:
                    json_obj = json.loads(line.strip())
                except json.JSONDecodeError:
                    print("Error decoding JSON in line:", line)
                context = json_obj['Context']
                question = json_obj['Question']
                answer = json_obj['Answer']
                system_content, user_content = split_input_text_link(context)
                message = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content + '\n' + question}
                ]
                reply = get_completion_from_messages(message)
                prediction_answer = get_matched_option(reply.lower(), options)

                if prediction_answer is not None:
                    right += int(prediction_answer.lower() == answer.lower())
                count += 1

        try:
            accuracy = right / count if count > 0 else 0
            print(f"accuracy: {right}/{count} ({accuracy:.2%})")
        except ZeroDivisionError as e:
            print(f"Error calculating accuracy: {e}")
            accuracy = 0
        #将结果写入log文件
        try:
            with open(f'results/{model_name}_log.txt', 'a') as f:
                f.write(f"{path} accuracy: {right}/{count} ({accuracy:.2%})\n")
        except Exception as e:
            print(f"Error writing to log file: {e}")



    except Exception as e:
        print(f"Error with combo {combo}: {e}", flush=True)

# 使用 ThreadPoolExecutor
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process dataset_name and case")
    parser.add_argument('--dataset_name', type=str, required=True, help="Name of the dataset")
    parser.add_argument('--case', type=int, required=True)

    args = parser.parse_args()

    dataset_name = args.dataset_name
    case = args.case

    dataset = dataset_name
    filtered_combinations = [{
        "dataset": dataset,
        "case": case
    }]
    combo = filtered_combinations[0]

    process_combination(combo)
