import os

from peft import PeftModel

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import json
from itertools import product
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
torch.cuda.empty_cache()

os.chdir('../../')
project_root = os.getcwd()
sys.path.append(project_root)

from LLM.utils import split_input_text, get_matched_option
from generate.node_classification_generate import get_data_and_text


# initialize Accelerator
if torch.cuda.is_bf16_supported():
    accelerator = Accelerator(mixed_precision="bf16")
    print('load bf16')
else:
    accelerator = Accelerator(mixed_precision="fp16")
    print('load fp16')
device = accelerator.device
print(device)


def process_combination(combinations):
    filtered_combinations = []
    for combo in combinations:
        dataset, mode, zero_shot_CoT, hop, include_label = combo

        if mode == "ego" and (include_label or hop != 1):
            continue

        filtered_combinations.append({
            "dataset": dataset,
            "mode": mode,
            "zero_shot_CoT": zero_shot_CoT,
            "hop": hop,
            "include_label": include_label
        })
    print(f"Number of combinations: {len(filtered_combinations)}", flush=True)
    return filtered_combinations



def get_completion_from_messages(messages, model, tokenizer, temperature=0.1, max_new_tokens=100):

    try:
        formatted_input = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([formatted_input], return_tensors="pt").to(model.device)

        outputs = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result
    except Exception as e:
        print(f"Error generating completion: {e}")
        return None



def run(combo):

    try:
        dataset, mode, zero_shot_CoT, hop, include_label = combo
        suffix = '.jsonl'
        if mode == 'pure structure':
            path = f'output/{dataset}/node_classification_pure_structure/test/'
        else:
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

        count = 0
        right = 0
        try:
            with open(path, 'r') as file:
                for line in tqdm(file, desc="Processing lines", unit="line"):
                    if count > 500:
                        break
                    try:
                        json_obj = json.loads(line.strip())
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in line: {line.strip()} - {e}")
                        continue

                    try:
                        context = json_obj['Context']
                        question = json_obj['Question']
                        answer = json_obj['Answer']
                    except KeyError as e:
                        print(f"Missing key in JSON object: {json_obj} - {e}")
                        continue

                    try:
                        system_content, user_content = split_input_text(context)
                        message = [
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": user_content + '\n' + question}
                        ]
                    except Exception as e:
                        print(f"Error in split_input_text or creating message: {e}")
                        continue

                    try:
                        if zero_shot_CoT:
                            reply = get_completion_from_messages(
                                message, model=model, tokenizer=tokenizer, max_new_tokens=200
                            )
                        else:
                            reply = get_completion_from_messages(
                                message, model=model, tokenizer=tokenizer)
                        if reply is None:
                            print("Error: Model did not generate a valid reply.")
                            continue
                        prediction_answer = get_matched_option(reply, options)
                        if prediction_answer is None:
                            print("Error: Could not match any option in the reply.")
                            continue
                    except Exception as e:
                        print(f"Error during model generation or matching answer: {e}")
                        continue

                    try:
                        if prediction_answer is not None:
                            right += int(prediction_answer.lower() == answer.lower())
                        count += 1
                    except Exception as e:
                        print(f"Error in accuracy calculation: {e}")
                        continue
        except FileNotFoundError as e:
            print(f"Test file not found at path {path}: {e}")
            return
        except Exception as e:
            print(f"Error reading file {path}: {e}")
            return

        try:
            accuracy = right / count if count > 0 else 0
            print(f"accuracy: {right}/{count} ({accuracy:.2%})")
        except ZeroDivisionError as e:
            print(f"Error calculating accuracy: {e}")
            accuracy = 0

        try:
            with open(f'results/{model_name}_tune_log.txt', 'a') as f:
                f.write(f"{path} accuracy: {right}/{count} ({accuracy:.2%})\n")
        except Exception as e:
            print(f"Error writing to log file: {e}")
    except Exception as e:
        print(f"Unhandled error with combo {combo}: {e}")

# 主函数
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--dataset_name', type=str, required=True, help="Name of the dataset")
    # parser.add_argument('--drop_rate', type=float, required=True, help="Drop rate for edge removal")
    parser.add_argument('--model_size', type=str, required=True, help="3B or 8B")
    parser.add_argument('--scenario', type=str, required=True,
                        help="no tune or 5 shot or 10 shot or pure structure or full fine-tune")
    parser.add_argument('--mode', type=str, required=True,
                       help="ego or neighbors")
    parser.add_argument('--hop', type=str, required=True, help="1 hop or 2 hop")
    parser.add_argument('--huggingface_token', type=str, required=True, help="huggingface_token")

    args = parser.parse_args()

    dataset = args.dataset_name
    model_size = args.model_size
    scenario = args.scenario
    mode = args.mode
    hop = 1 if args.case == '1 hop' else 2
    huggingface_token = args.huggingface_token

    if scenario == 'pure structure':
        mode = 'pure structure'

    # 模型配置
    model_id = 'meta-llama/Llama-3.2-3B-Instruct' if model_size == '3B' else 'meta-llama/Llama-3.1-8B-Instruct'
    model_name = f'llama_{model_size}_Instruct_{dataset}_node'
    if scenario == 'no tune':
        model_name += '_no_tune'
    elif scenario == '5 shot':
        model_name += '_5shot'
    elif scenario == '10 shot':
        model_name += '_10shot'
    elif scenario == 'pure structure':
        model_name += '_pure_structure'
    elif scenario == 'full fine-tune':
        pass
    else:
        raise ValueError("Invalid scenario")

    import glob

    lora_model_name_or_path = f'output_model/{model_name}/checkpoint*'
    checkpoint_paths = glob.glob(lora_model_name_or_path)

    if checkpoint_paths:
        lora_model_name_or_path = checkpoint_paths[0]
        print(f"Using checkpoint: {lora_model_name_or_path}")
    else:
        print("No checkpoint found.")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=huggingface_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    offload_folder = "./offload"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=huggingface_token,
        offload_folder=offload_folder,
        trust_remote_code=True
    )
    # model = AutoModelForCausalLM.from_pretrained(model_id, token=huggingface_token,trust_remote_code=True)
    if scenario != 'no tune':
        model = PeftModel.from_pretrained(model, lora_model_name_or_path)
    model.eval()

    params = {
        "dataset_name": [dataset],
        # "mode": ["ego", "pure structure","neighbors"],
        "mode": [mode],
        "zero_shot_CoT": [False],
        # "zero_shot_CoT": [True],
        "hop": [hop],
        "include_label": [False]
    }

    combinations = list(product(
        params["dataset_name"],
        params["mode"],
        params["zero_shot_CoT"],
        params["hop"],
        params["include_label"]
    ))

    filtered_combinations = process_combination(combinations)



    try:
        dataloader = DataLoader(filtered_combinations, batch_size=1)
        print(f"Dataset length: {len(dataloader.dataset)}")
        print(f"Batch size: {dataloader.batch_size}")

        try:
            model, dataloader = accelerator.prepare(model, dataloader)
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model = model.module

            model = accelerator.unwrap_model(model)
            print(f"Model and DataLoader successfully prepared on {device}")
        except Exception as e:
            print(f"Error preparing model or dataloader: {e}")
            sys.exit(1)

        for i, batch in enumerate(tqdm(dataloader, desc="Processing combinations")):
            try:
                # 解析 batch
                dataset = batch['dataset'][0]
                mode = batch['mode'][0]
                zero_shot_CoT = batch['zero_shot_CoT'][0].item()
                hop = batch['hop'][0].item()
                include_label = batch['include_label'][0].item()

                combo = (dataset, mode, zero_shot_CoT, hop, include_label)
                print(f"Combo {i}: {combo}")

                run(combo)
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error processing batch {i}: {e}")

    except Exception as e:
        print(f"Error during processing: {e}")