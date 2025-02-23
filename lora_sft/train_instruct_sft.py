import os
import sys
os.chdir('../')
print(os.getcwd())
project_root = os.getcwd()
sys.path.append(project_root)

from LLM.utils import split_input_text, split_input_text_link
from peft.tuners.lora import LoraLayer
import copy
import json
import logging
import logging
import torch
import transformers
from datasets import Dataset
from dataclasses import dataclass, field
from datasets import load_dataset
from functools import partial
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq, Trainer
from typing import Dict, Optional, Sequence, List

logger = logging.getLogger(__name__)
huggingface_token = 'your token'

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_lora: Optional[bool] = field(default=False)
    save_path: Optional[str] = field(default="unknown")


@dataclass
class DataArguments:
    # data_path: str = field(default=None, metadata={
    #     "help": "Path to the training data."})
    data_path: List[str] = field(default_factory=list, metadata={
        "help": "List of paths to the training data."})
    source_length: int = field(default=512)
    target_length: int = field(default=512)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_deepspeed: bool = field(default=False)


def load_jsonl_files(data_path: list[str]) -> Dataset:
    """
    directly load JSONL file and transform it to Huggingface Dataset object

    """
    def get_all_datapath(root_dirs):
        train_files = []
        for root_dir in root_dirs:
            for root, dirs, files in os.walk(root_dir):
                if os.path.basename(root) == "train":
                    for file in files:
                        if 'case' in file and 'case0' not in file and 'case1' not in file:
                            continue
                        if 'with_label' in file:
                            continue
                        file_path = os.path.join(root, file)
                        print(file_path)
                        train_files.append(file_path)
        return train_files

    all_file_list = get_all_datapath(data_path)

    data = []
    for file_path in tqdm(all_file_list, desc="Loading JSONL files"):
        if 'node' in file_path:
            print("data_type: node")
            data_process_func = split_input_text
        else:
            print("data_type: link")
            data_process_func = split_input_text_link

        count = 0
        with open(file_path, 'r') as f:
            for line in f:
                if count > 5000 and 'product' in file_path and 'node_classification' in file_path:
                    break
                if count > 5000 and 'product' in file_path and 'link_prediction' in file_path:
                    break
                count += 1
                try:
                    record = json.loads(line.strip())
                    context = record['Context']
                    question = record['Question']
                    answer = record['Answer']
                    system_content, user_content = data_process_func(context)
                    user_content = user_content + '\n' + question
                    record = {
                        "system_content": system_content,
                        "user_content": user_content,
                        "answer": str(answer)
                    }
                    data.append(record)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {file_path}: {e}")


    print(f"Loaded {len(data)} records from {len(all_file_list)} files.")

    import datasets
    dataset = datasets.Dataset.from_list(data)
    return dataset


IGNORE_INDEX = -100


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0]
                          for tokenized in tokenized_list]
    ne_pad_token_id = IGNORE_INDEX if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(ne_pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(
        strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def make_train_dataset(tokenizer: transformers.PreTrainedTokenizer, data_path: list[str], data_args: DataArguments) -> Dataset:
    logging.warning("Loading data...")

    dataset = load_jsonl_files(data_path=data_path)
    logging.warning("Formatting inputs...")
    # prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

    def generate_sources_targets(examples: Dict, tokenizer: transformers.PreTrainedTokenizer):
        system_content = examples['system_content']
        user_content = examples['user_content']
        answer = examples['answer']

        sources = []
        for i in range(len(system_content)):
            message = [
                {"role": "system", "content": system_content[i]},
                {"role": "user", "content": user_content[i]}
            ]
            formatted_input = tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            sources.append(formatted_input)
            # model_inputs = tokenizer([formatted_input], return_tensors="pt").to(model.device)

        targets = [f"{example}{tokenizer.eos_token}" for example in answer]


        input_output = preprocess(
            sources=sources, targets=targets, tokenizer=tokenizer)
        examples['input_ids'] = input_output['input_ids']
        examples['labels'] = input_output['labels']
        return examples


    generate_sources_targets_p = partial(
        generate_sources_targets, tokenizer=tokenizer)

    dataset = dataset.map(
        function=generate_sources_targets_p,
        batched=True,
        desc="Running tokenizer on train dataset",
        num_proc=20
    ).shuffle()
    return dataset



def load_model_and_tokenizer(model_args: ModelArguments, training_args: TrainingArguments, data_args: DataArguments) -> tuple:

    if training_args.use_deepspeed:

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            # cache_dir=training_args.cache_dir,
            torch_dtype='auto',
            # if model_args.model_name_or_path.find("falcon") != -1 else False
            trust_remote_code=True,
            token=huggingface_token

        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            # cache_dir=training_args.cache_dir,
            device_map='auto',
            torch_dtype='auto',
            # if model_args.model_name_or_path.find("falcon") != -1 else False
            trust_remote_code=True,
            token=huggingface_token

        )

    if model_args.use_lora:

        logging.warning("Loading model to Lora")

        from peft import LoraConfig, get_peft_model
        LORA_R = 16
        LORA_ALPHA = 32
        LORA_DROPOUT = 0.05
        TARGET_MODULES = [
            "o_proj","gate_proj", "down_proj", "up_proj"
        ]

        config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        # model = model.to(torch.bfloat16)
        model = get_peft_model(model, config)
        # peft_module_casting_to_bf16(model)
        model.print_trainable_parameters()

    # model.is_parallelizable = True
    # model.model_parallel = True
    # torch.cuda.empty_cache()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True, token=huggingface_token)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model, tokenizer = load_model_and_tokenizer(
        model_args, training_args, data_args)

    with training_args.main_process_first(desc="loading and tokenization"):

        train_dataset = make_train_dataset(
            tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model,
                                           label_pad_token_id=IGNORE_INDEX
                                           )

    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=None,
                      data_collator=data_collator)
    trainer.train()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    train()
