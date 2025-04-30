import copy

import torch
import numpy as np
import json
import os
import random
import openai
import sys

from scipy.ndimage import label
from tqdm import tqdm

from utils.load_arxiv import get_raw_text_arxiv
from utils.load_cora import get_raw_text_cora
from utils.load_pubmed import get_raw_text_pubmed
from utils.load_arxiv_2023 import get_raw_text_arxiv_2023
from utils.load_products import get_raw_text_products
from utils.load_heg import get_raw_text_heg
from time import sleep
from utils.prompts import generate_system_prompt, arxiv_natural_lang_mapping, generate_question_prompt

from time import sleep
from random import randint, sample
import threading


#
qwen_api_key = "your_key"
model="qwen-plus"
max_tokens=500
# openai.api_key  = qwen_api_key

def load_data(dataset, trian_perc, val_perc, test_perc, use_text=False, seed=0):
    """
    Load data based on the dataset name.

    Parameters:
        dataset (str): Name of the dataset to be loaded. Options are "cora", "pubmed", "arxiv", "arxiv_2023", and "product".
        use_text (bool, optional): Whether to use text data. Default is False.
        seed (int, optional): Random seed for data loading. Default is 0.

    Returns:
        Tuple: Loaded data and text information.

    Raises:
        ValueError: If the dataset name is not recognized.
    """

    if dataset == "cora":
        data, text = get_raw_text_cora(trian_perc, val_perc, test_perc, use_text, seed)
    elif dataset == "pubmed":
        data, text = get_raw_text_pubmed(trian_perc, val_perc, test_perc, use_text, seed)
    elif dataset == "arxiv":
        data, text = get_raw_text_arxiv(use_text)
    elif dataset == "arxiv_2023":
        data, text = get_raw_text_arxiv_2023(use_text)
    elif dataset == "product":
        data, text = get_raw_text_products(use_text)
    elif dataset in ['Actor', 'Amazon']:
        data, text = get_raw_text_heg(dataset, use_text)
    else:
        raise ValueError("Dataset must be one of: cora, pubmed, arxiv")
    return data, text


def get_subgraph(node_idx, edge_index, hop=1):
    """
    Get subgraph around a specific node up to a certain hop.

    Parameters:
        node_idx (int): Index of the node.
        edge_index (torch.Tensor): Edge index tensor.
        hop (int, optional): Number of hops around the node to consider. Default is 1.

    Returns:
        list: Lists of nodes for each hop distance.
    """

    current_nodes = torch.tensor([node_idx])
    all_hops = []

    for _ in range(hop):
        mask = torch.isin(edge_index[0], current_nodes) | torch.isin(edge_index[1], current_nodes)
        
        # Add both the source and target nodes involved in the edges 
        new_nodes = torch.unique(torch.cat((edge_index[0][mask], edge_index[1][mask])))

        # Remove the current nodes to get only the new nodes added in this hop
        diff_nodes_set = set(new_nodes.numpy()) - set(current_nodes.numpy())
        diff_nodes = torch.tensor(list(diff_nodes_set))  
        
        all_hops.append(diff_nodes.tolist())

        # Update current nodes for the next iteration
        current_nodes = torch.unique(torch.cat((current_nodes, new_nodes)))

    return all_hops



def get_and_save_message_for_node(node_index_list, data, text, dataset, source, save_dir, hop=2,
                        max_papers_1=20, max_papers_2=10, mode="ego",
                        include_label=False, abstract_len=0, arxiv_style=False,
                        include_options=False, include_abs=False, zero_shot_CoT=False, BAG=False,
                        few_shot=False, options=None, is_train=True):
    if mode == 'pure structure':
        path = save_dir + f"/{dataset}" + '/node_classification_pure_structure'
    else:
        path = save_dir + f"/{dataset}" + '/node_classification'
    dir = 'train' if is_train else 'test'
    has_label = 'with_label' if include_label else 'without_label'
    for node_index in tqdm(node_index_list, desc="Processing nodes"):
        if mode == 'neighbors':
            prefix_prompt = ('You are a good graph reasoner. '
                             f'Give you a graph language that describes a graph structure and node information from {source} dataset. '
                             'You need to understand the graph and the task definition and answer the question.\n')

            context = "" + prefix_prompt
            context += '\n## Target node:\n'
            if source == 'product' or source == 'Amazon':
                context += f'Product id: {node_index}\n'
            elif source == 'Actor':
                context += f'Actor id: {node_index}\n'
            else:
                context += f'Paper id: {node_index}\n'
            question = generate_question_prompt(source, arxiv_style, include_options, is_train)
            answer = text['label'][node_index]

            if source in ['Actor', 'Amazon']:
                content = text['content'][node_index]
                context = f"{context}Content: {content}\n"
            else:
                title = text['title'][node_index]
                if source == 'product':
                    content = text['content'][node_index]
                    if include_abs:
                        context = f"{context}Title: {title}\nContent: {content}\n"
                    else:
                        context = f"{context}Title: {title}\n"
                else:
                    abstract = text['abs'][node_index]
                    if include_abs:
                        context = f"{context}Title: {title}\nAbstract: {abstract}\n"
                    else:
                        context = f"{context}Title: {title}\n"


            # all_hops = get_hop_nodes(data, node_index, data.edge_index, is_train, hop)
            all_hops = get_subgraph(node_index, data.edge_index, hop)
            prompt_str = generate_structure_prompt(node_index, text, all_hops, data, hop, max_papers_1,
                                                        max_papers_2, abstract_len, include_label, dataset, is_train, mode)
            context += prompt_str

            json_data = {
                "Context": context,
                "Question": question,
                "Answer": answer
            }

            output_file = path + f'/{dir}/{hop}_hop_{has_label}.jsonl'
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "a") as f:
                f.write(json.dumps(json_data) + "\n")

        elif mode == 'ego':
            prefix_prompt = ('You are a good graph reasoner. '
                             f'Give you a graph language that describes the target node information from {source} dataset. '
                             'You need to understand the graph and the task definition and answer the question.\n')
            context = "" + prefix_prompt
            context += '\n## Target node:\n'
            if source == 'product':
                context += f'Product id: {node_index}\n'
            else:
                context += f'Paper id: {node_index}\n'
            question = generate_question_prompt(source, arxiv_style, include_options, is_train, zero_shot_CoT, BAG, few_shot)
            answer = text['label'][node_index]

            title = text['title'][node_index]
            if source == 'product':
                content = text['content'][node_index]
                if include_abs:
                    context = f"{context}Title: {title}\nContent: {content}\n"
                else:
                    context = f"{context}Title: {title}\n"
            else:
                abstract = text['abs'][node_index]
                if include_abs:
                    context = f"{context}Title: {title}\nAbstract: {abstract}\n"
                else:
                    context = f"{context}Title: {title}\n"

            json_data = {
                "Context": context,
                "Question": question,
                "Answer": answer
            }

            if zero_shot_CoT and not is_train:
                output_file= path + f"/{dir}/ego_CoT.jsonl"
            elif BAG and not is_train:
                output_file= path + f"/{dir}/ego_BAG.jsonl"
            elif few_shot and not is_train:
                output_file= path + f"/{dir}/ego_few_shot.jsonl"
            else:
                output_file= path + f"/{dir}/ego.jsonl"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "a") as f:
                f.write(json.dumps(json_data) + "\n")

        elif mode == 'pure structure':
            prefix_prompt = ('You are a good graph reasoner. '
                             f'Give you a graph language that describes a graph structure from {source} dataset. '
                             'You need to understand the graph and the task definition and answer the question.\n')

            context = "" + prefix_prompt
            context += '\n## Target node:\n'
            if source == 'product':
                context += f'Product id: {node_index}\n'
            else:
                context += f'Paper id: {node_index}\n'
            question = generate_question_prompt(source, arxiv_style, include_options, is_train)
            answer = text['label'][node_index]

            if source == "product":
                max_papers_1 = 30
                max_papers_2 = 60
            else:
                max_papers_1 = 20
                max_papers_2 = 40
            # all_hops = get_hop_nodes(data, node_index, data.edge_index, is_train, hop)
            all_hops = get_subgraph(node_index, data.edge_index, hop)
            prompt_str = generate_structure_prompt(node_index, text, all_hops, data, hop, max_papers_1,
                                                   max_papers_2, abstract_len, include_label, dataset, is_train, mode)
            context += prompt_str

            json_data = {
                "Context": context,
                "Question": question,
                "Answer": answer
            }


            output_file = path + f'/{dir}/pure_structure_{hop}_hop_{has_label}.jsonl'
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "a") as f:
                f.write(json.dumps(json_data) + "\n")

        else:
            print('Invalid mode! Please use either "neighbors" or "abstract"')
    print(output_file + ' has completed!')

def get_and_save_message_for_link(node_index_list, data, text, dataset, source, save_dir, max_papers_1=10,
                                             max_papers_2=20, max_papers_3=5, options=None, is_train=True,
                                             include_title=True, case=1):
    if include_title:
        path = save_dir + f"/{dataset}" + '/link_prediction'
    else:
        path = save_dir + f"/{dataset}" + '/link_prediction_no_title'
    dir = 'train' if is_train else 'test'
    if include_title:
        if case != 0 and case != 1:

            # prefix_prompt = ('You are a good graph reasoner. '
            #                  f'Give you a graph language that describes a graph structure and node information from {source} dataset. '
            #                  'You need to understand the graph and the task definition and answer the question.\n')
            prefix_prompt = ('You are a good graph reasoner. '
                             f'Give you a graph language that describes a graph structure and node information from {source} dataset. '
                             'You need to understand the graph and answer the question. When you make a decision, please carefully consider the graph structure and the node information.\n')
            # prefix_prompt = (f'You are a graph reasoner. Based on the {source} dataset, determine whether two target nodes are connected by an edge, using the following rules:'
            #                   '\n1. An edge exists if the two nodes share at least one direct neighbor.'
            #                   '\n2. If no direct neighbor is shared, evaluate semantic similarity between the nodes\' titles or their neighbors\' titles to infer a connection.'
            #                   '\n3. Only if no direct neighbor exists and no semantic similarity is detected, answer "No"')
        else:
            prefix_prompt = (
                f'You are a good graph reasoner. Based on the {source} dataset, determine whether two target nodes are connected by an edge. When you make a decision, please carefully consider the graph structure and the node information. If two nodes share similar structure or information, they are likely to be connected.\n')


    else:
        if case != 0 and case != 1:
            prefix_prompt = ('You are a good graph reasoner. '
                             f'Give you a graph language that describes a graph structure and node information from {source} dataset. '
                             'You need to understand the graph and answer the question. When you make a decision, please carefully consider the graph structure and the node information.\n')
        else:
            prefix_prompt = (
                f'You are a good graph reasoner. Based on the {source} dataset, determine whether two target nodes are connected by an edge. When you make a decision, please carefully consider the graph structure and the node information. If two nodes share similar structure or information, they are likely to be connected.\n')


    for node_index in tqdm(node_index_list, desc="Processing links"):
        context = "" + prefix_prompt
        context += "## Target node1:\n\n"
        if source == 'product':
            context += f'Product id: {node_index}\n'
        else:
            context += f'Paper id: {node_index}\n'

        if include_title:
            title = text['title'][node_index]
            context += f"Title: {title}"


        if case == 0:
            all_hops = get_subgraph(node_index, data.edge_index, hop=1)
            all_hops = filter_train_or_test_nodes(data, all_hops, is_train)
            if all_hops[0] == []:
                continue


            question = '\n\nAre Target Node1 and Target Node2 connected? Do not provide your reasoning. Only provide "Yes" or "No" based on the rules above.\n Answer:'
            question = '\n\nAre Target Node1 and Target Node2 connected? Do not provide your reasoning. Only provide "Yes" or "No" based on your inference.\n Answer:'


            #----------right node------------
            right_node_index = random.choice(all_hops[0])
            all_hops[0].remove(right_node_index)
            prompt_str = generate_structure_prompt_for_link(node_index, text, all_hops, data, max_papers_1,
                                                            max_papers_2, dataset, include_title, hop=1)
            context += prompt_str

            temp_context = ''
            temp_context += "\n## Target node2:\n\n"
            target_node_2_index = right_node_index
            if source == 'product':
                temp_context += f'Product id: {target_node_2_index}\n'
            else:
                temp_context += f'Paper id: {target_node_2_index}\n'
            if include_title:
                title = text['title'][target_node_2_index]
                temp_context += f"Title: {title}\n"

            target_node_2_all_hops = get_subgraph(target_node_2_index, data.edge_index, hop=1)
            target_node_2_all_hops = filter_train_or_test_nodes(data, target_node_2_all_hops, is_train)
            if node_index in target_node_2_all_hops[0]:
                target_node_2_all_hops[0].remove(node_index)
            prompt_str = generate_structure_prompt_for_link(node_index, text, target_node_2_all_hops, data,
                                                            max_papers_1,
                                                            max_papers_2, dataset, include_title, hop=1)
            temp_context += prompt_str

            answer = 'Yes'
            json_data = {
                "Context": context + temp_context,
                "Question": question,
                "Answer": answer
            }
            output_file = path + f'/{dir}/case0.jsonl'
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "a") as f:
                f.write(json.dumps(json_data) + "\n")

            # ----------wrong node------------
            while True:
                sample_node = random.choice(node_index_list)
                if sample_node != node_index and sample_node not in all_hops[0]:
                    target_node_2_index = sample_node
                    break
            temp_context = ''
            temp_context += "\n## Target node2:\n\n"
            if source == 'product':
                temp_context += f'Product id: {target_node_2_index}\n'
            else:
                temp_context += f'Paper id: {target_node_2_index}\n'
            if include_title:
                title = text['title'][target_node_2_index]
                temp_context += f"Title: {title}\n"

            target_node_2_all_hops = get_subgraph(target_node_2_index, data.edge_index, hop=1)
            target_node_2_all_hops = filter_train_or_test_nodes(data, target_node_2_all_hops, is_train)
            prompt_str = generate_structure_prompt_for_link(node_index, text, target_node_2_all_hops, data,
                                                            max_papers_1,
                                                            max_papers_2, dataset, include_title, hop=1)
            temp_context += prompt_str
            answer = 'No'
            json_data = {
                "Context": context + temp_context,
                "Question": question,
                "Answer": answer
            }
            output_file = path + f'/{dir}/case0.jsonl'
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "a") as f:
                f.write(json.dumps(json_data) + "\n")


        elif case == 1:
            all_hops = get_subgraph(node_index, data.edge_index, hop=2)
            all_hops = filter_train_or_test_nodes(data, all_hops, is_train)
            if all_hops[0] == []:
                continue

            # question = '\n\nBased on the available partial information. Can Target node1 be connected with Target node2? '
            # question += '\nDo not provide your reasoning. The answer should be Yes or No\n Answer:\n\n'
            question = '\n\nAre Target Node1 and Target Node2 connected? Do not provide your reasoning. Only provide "Yes" or "No" based on the rules above.\n Answer:'
            question = '\n\nAre Target Node1 and Target Node2 connected? Do not provide your reasoning. Only provide "Yes" or "No" based on your inference.\n Answer:'


            #----------right node------------
            right_node_index = random.choice(all_hops[0])
            all_hops[0].remove(right_node_index)
            prompt_str = generate_structure_prompt_for_link(node_index, text, all_hops, data, max_papers_1,
                                                            max_papers_2, dataset, include_title, hop=2)
            context += prompt_str

            temp_context = ''
            temp_context += "\n## Target node2:\n\n"
            target_node_2_index = right_node_index
            if source == 'product':
                temp_context += f'Product id: {target_node_2_index}\n'
            else:
                temp_context += f'Paper id: {target_node_2_index}\n'
            if include_title:
                title = text['title'][target_node_2_index]
                temp_context += f"Title: {title}\n"

            target_node_2_all_hops = get_subgraph(target_node_2_index, data.edge_index, hop=2)
            target_node_2_all_hops = filter_train_or_test_nodes(data, target_node_2_all_hops, is_train)
            if node_index in target_node_2_all_hops[0]:
                target_node_2_all_hops[0].remove(node_index)
            prompt_str = generate_structure_prompt_for_link(node_index, text, target_node_2_all_hops, data,
                                                            max_papers_1,
                                                            max_papers_2, dataset, include_title, hop=2)
            temp_context += prompt_str

            answer = 'Yes'
            json_data = {
                "Context": context + temp_context,
                "Question": question,
                "Answer": answer
            }
            output_file = path + f'/{dir}/case1.jsonl'
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "a") as f:
                f.write(json.dumps(json_data) + "\n")

            # ----------wrong node------------
            while True:
                sample_node = random.choice(node_index_list)
                if sample_node != node_index and sample_node not in all_hops[0]:
                    target_node_2_index = sample_node
                    break
            temp_context = ''
            temp_context += "\n## Target node2:\n\n"
            if source == 'product':
                temp_context += f'Product id: {target_node_2_index}\n'
            else:
                temp_context += f'Paper id: {target_node_2_index}\n'
            if include_title:
                title = text['title'][target_node_2_index]
                temp_context += f"Title: {title}\n"

            target_node_2_all_hops = get_subgraph(target_node_2_index, data.edge_index, hop=2)
            target_node_2_all_hops = filter_train_or_test_nodes(data, target_node_2_all_hops, is_train)
            prompt_str = generate_structure_prompt_for_link(node_index, text, target_node_2_all_hops, data,
                                                            max_papers_1,
                                                            max_papers_2, dataset, include_title, hop=2)
            temp_context += prompt_str
            answer = 'No'
            json_data = {
                "Context": context + temp_context,
                "Question": question,
                "Answer": answer
            }
            output_file = path + f'/{dir}/case1.jsonl'
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "a") as f:
                f.write(json.dumps(json_data) + "\n")



        elif case == 2:
            all_hops = get_subgraph(node_index, data.edge_index, hop=2)
            all_hops = filter_train_or_test_nodes(data, all_hops, is_train)
            if all_hops[0] == [] or all_hops[1] == []:
                continue
            # all_hops = filter_train_or_test_nodes(data, all_hops, is_train)
            sampled_values, updated_hops = sample_and_remove(all_hops)
            prompt_str = generate_structure_prompt_for_link(node_index, text, updated_hops, data, max_papers_1,
                                                   max_papers_2, dataset, include_title, hop=1)

            context += prompt_str
            count = 0
            for sample_index in sampled_values:
                temp_context = ''
                temp_context += "\n## Target node2:\n\n"
                target_node_2_index = sample_index
                if source == 'product':
                    temp_context += f'Product id: {target_node_2_index}\n'
                else:
                    temp_context += f'Paper id: {target_node_2_index}\n'
                if include_title:
                    title = text['title'][target_node_2_index]
                    temp_context += f"Title: {title}\n"

                target_node_2_all_hops = get_subgraph(target_node_2_index, data.edge_index, hop=1)
                target_node_2_all_hops = filter_train_or_test_nodes(data, target_node_2_all_hops, is_train)
                if node_index in target_node_2_all_hops[0]:
                    target_node_2_all_hops[0].remove(node_index)
                prompt_str = generate_structure_prompt_for_link(node_index, text, target_node_2_all_hops, data, max_papers_1,
                                                                max_papers_2, dataset, include_title, hop=1)
                temp_context += prompt_str

                question = '\n\nBased on the available partial information. Can Target node1 be connected with Target node2? '
                question += '\nDo not provide your reasoning. The answer should be Yes or No\n Answer:\n\n'

                if count == 0:
                    answer = "Yes"
                else:
                    answer = "No"

                json_data = {
                    "Context": context + temp_context,
                    "Question": question,
                    "Answer": answer
                }

                output_file = path + f'/{dir}/case2.jsonl'
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, "a") as f:
                    f.write(json.dumps(json_data) + "\n")
                count += 1

        elif case == 3:
            all_hops = get_subgraph(node_index, data.edge_index, hop=3)
            all_hops = filter_train_or_test_nodes(data, all_hops, is_train)
            if all_hops[0] == [] or all_hops[1] == [] or all_hops[2] == []:
                continue
            sampled_values, updated_hops = sample_and_remove(all_hops)
            prompt_str = generate_structure_prompt_for_link(node_index, text, updated_hops, data, max_papers_1,
                                                            max_papers_2, dataset, include_title, hop=2)
            context += prompt_str

            question = '\n\nBased on the available partial information. Can Target node2 be a 2-hop neighbor of Target node1? '
            question += '\nDo not provide your reasoning. The answer should be Yes or No\n Answer:\n\n'

            count = 0
            for sample_index in sampled_values:
                if count == 0:
                    count += 1
                    continue
                temp_context = ''
                temp_context += "\n## Target node2:\n\n"
                target_node_2_index = sample_index
                if source == 'product':
                    temp_context += f'Product id: {target_node_2_index}\n'
                else:
                    temp_context += f'Paper id: {target_node_2_index}\n'
                if include_title:
                    title = text['title'][target_node_2_index]
                    temp_context += f"Title: {title}\n"

                if count == 1:
                    answer = "Yes"
                else:
                    answer = "No"

                json_data = {
                    "Context": context + temp_context,
                    "Question": question,
                    "Answer": answer
                }

                output_file = path + f'/{dir}/case3.jsonl'
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, "a") as f:
                    f.write(json.dumps(json_data) + "\n")
                count += 1

        elif case == 4:
            all_hops = get_subgraph(node_index, data.edge_index, hop=2)
            original_hops = copy.deepcopy(all_hops)
            all_hops = filter_train_or_test_nodes(data, all_hops, is_train)
            if all_hops[0] == [] or all_hops[1] == []:
                continue
            prompt_str = generate_structure_prompt_for_link(node_index, text, all_hops, data, max_papers_1,
                                                            max_papers_2, dataset, include_title, hop=1)
            context += prompt_str

            question = '\n\nBased on the available partial information. Can Target node1 be connected with Target node2 through the Middle node? '
            question += '\nDo not provide your reasoning. The answer should be Yes or No\n Answer:\n\n'

            times = min(3, len(all_hops[1]))
            for i in range(times):
                temp_context = ''
                temp_context += "\n## Target node2:\n\n"
                target_node_2_index = all_hops[1][i]
                if source == 'product':
                    temp_context += f'Product id: {target_node_2_index}\n'
                else:
                    temp_context += f'Paper id: {target_node_2_index}\n'
                if include_title:
                    title = text['title'][target_node_2_index]
                    temp_context += f"Title: {title}\n"

                right_middle_nodes, wrong_middle_nodes = get_right_wrong_middle_nodes(original_hops, target_node_2_index, data.edge_index)
                middle_node_context = ''
                if len(right_middle_nodes) == 0:
                    continue
                else:
                    middle_node_context += "\n## Middle node:\n\n"
                    middle_node_index = right_middle_nodes[0]
                    if source == 'product':
                        middle_node_context += f'Product id: {middle_node_index}\n'
                    else:
                        middle_node_context += f'Paper id: {middle_node_index}\n'
                    if include_title:
                        title = text['title'][middle_node_index]
                        middle_node_context += f"Title: {title}\n"

                    answer = 'Yes'
                    json_data = {
                        "Context": context + temp_context + middle_node_context,
                        "Question": question,
                        "Answer": answer
                    }

                    output_file = path + f'/{dir}/case4.jsonl'
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    with open(output_file, "a") as f:
                        f.write(json.dumps(json_data) + "\n")

                middle_node_context = ''
                if len(wrong_middle_nodes) == 0:
                    continue
                else:
                    middle_node_context += "\n## Middle node:\n\n"
                    middle_node_index = wrong_middle_nodes[0]
                    if source == 'product':
                        middle_node_context += f'Product id: {middle_node_index}\n'
                    else:
                        middle_node_context += f'Paper id: {middle_node_index}\n'
                    if include_title:
                        title = text['title'][middle_node_index]
                        middle_node_context += f"Title: {title}\n"

                    answer = 'No'
                    json_data = {
                        "Context": context + temp_context + middle_node_context,
                        "Question": question,
                        "Answer": answer
                    }

                    output_file = path + f'/{dir}/case4.jsonl'
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    with open(output_file, "a") as f:
                        f.write(json.dumps(json_data) + "\n")

        elif case == 5:
            all_hops = get_subgraph(node_index, data.edge_index, hop=4)
            all_hops = filter_train_or_test_nodes(data, all_hops, is_train)
            if all_hops[0] == [] or all_hops[1] == [] or all_hops[2] == [] or all_hops[3] == []:
                continue
            sampled_values, updated_hops = sample_and_remove(all_hops)
            prompt_str = generate_structure_prompt_for_link(node_index, text, all_hops, data, max_papers_1,
                                                            max_papers_2, dataset, include_title, hop=2)
            context += prompt_str

            question = '\n\nBased on the available partial information. Can Target node2 be a 3-hop neighbor of Target node1? '
            question += '\nDo not provide your reasoning. The answer should be Yes or No\n Answer:\n\n'


            for i in range(len(sampled_values)):
                if i < 2:
                    continue
                temp_context = ''
                temp_context += "\n## Target node2:\n\n"
                target_node_2_index = sampled_values[i]
                if source == 'product':
                    temp_context += f'Product id: {target_node_2_index}\n'
                else:
                    temp_context += f'Paper id: {target_node_2_index}\n'
                if include_title:
                    title = text['title'][target_node_2_index]
                    temp_context += f"Title: {title}"

                target_node_2_all_hops = get_subgraph(target_node_2_index, data.edge_index, hop=1)
                target_node_2_all_hops = filter_train_or_test_nodes(data, target_node_2_all_hops, is_train)
                prompt_str = generate_structure_prompt_for_link(node_index, text, target_node_2_all_hops, data, max_papers_1,
                                                                max_papers_2, dataset, include_title, hop=1)
                temp_context += prompt_str


                if i == 2:
                    answer = "Yes"
                else:
                    answer = "No"

                json_data = {
                    "Context": context + temp_context,
                    "Question": question,
                    "Answer": answer
                }

                output_file = path + f'/{dir}/case5.jsonl'
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, "a") as f:
                    f.write(json.dumps(json_data) + "\n")

        elif case == 6:
            all_hops = get_subgraph(node_index, data.edge_index, hop=1)
            all_hops = filter_train_or_test_nodes(data, all_hops, is_train)
            if all_hops[0] == []:
                continue
            sampled_node = random.choice(all_hops[0])
            all_hops[0].remove(sampled_node)
            prompt_str = generate_structure_prompt_for_link(node_index, text, all_hops, data, max_papers_1,
                                                            max_papers_2, dataset, include_title, hop=1)
            context += prompt_str

            question = '\n\nBased on the available partial information. Which other node will be connected to Target node1 within one hop? '
            question += '\nDo not provide your reasoning. The answer should be the paper id.\n Answer:\n\n'

            answer = str(sampled_node)

            json_data = {
                "Context": context,
                "Question": question,
                "Answer": answer
            }

            output_file = path + f'/{dir}/case6.jsonl'
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "a") as f:
                f.write(json.dumps(json_data) + "\n")

        elif case == 7:
            all_hops = get_subgraph(node_index, data.edge_index, hop=3)
            all_hops = filter_train_or_test_nodes(data, all_hops, is_train)
            if all_hops[0] == []:
                continue
            node_pool = sum(all_hops[1:], [])
            if len(node_pool) < 3:
                continue
            one_hop_node = random.choice(all_hops[0])
            all_hops[0].remove(one_hop_node)
            prompt_str = generate_structure_prompt_for_link(node_index, text, all_hops, data, max_papers_1,
                                                           max_papers_2, dataset, include_title, hop=1)
            context += prompt_str

            sample_nodes = np.random.choice(node_pool, size=3, replace=False).tolist()
            sample_nodes.append(one_hop_node)
            choice_list = []
            for i in range(len(sample_nodes)):
                temp_context = ''
                temp_node = sample_nodes[i]
                if source == 'product':
                    temp_context += f'Product id: {temp_node} '
                else:
                    temp_context += f'Paper id: {temp_node} '
                if include_title:
                    title = text['title'][temp_node]
                    temp_context += f"Title: {title}\n"
                choice_list.append(temp_context)
            random.shuffle(choice_list)

            right_loc = 0
            for i in range(len(choice_list)):
                if choice_list[i].find(f'{one_hop_node}') != -1:
                    right_loc = i
                    break
            sign = ['A', 'B', 'C', 'D']
            question = '\n\nBased on the available partial information. Which other node can be connected to Target node1 within one hop?\n '
            for i in range(4):
                question += f'{sign[i]}. {choice_list[i]}'
            question += '\nDo not provide your reasoning. The answer should be A, B, C or D.\n Answer:\n\n'

            answer = sign[right_loc]

            json_data = {
                "Context": context,
                "Question": question,
                "Answer": answer
            }

            output_file = path + f'/{dir}/case7.jsonl'
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "a") as f:
                f.write(json.dumps(json_data) + "\n")

        elif case == 8:
            all_hops = get_subgraph(node_index, data.edge_index, hop=3)
            all_hops = filter_train_or_test_nodes(data, all_hops, is_train)
            if all_hops[1] == []:
                continue
            node_pool = all_hops[0] + all_hops[2]
            if len(node_pool) < 3:
                continue
            one_hop_node = random.choice(all_hops[1])
            all_hops[1].remove(one_hop_node)
            prompt_str = generate_structure_prompt_for_link(node_index, text, all_hops, data, max_papers_1,
                                                            max_papers_2, dataset, include_title, hop=2)
            context += prompt_str

            sample_nodes = np.random.choice(node_pool, size=3, replace=False).tolist()
            sample_nodes.append(one_hop_node)
            choice_list = []
            for i in range(len(sample_nodes)):
                temp_context = ''
                temp_node = sample_nodes[i]
                if source == 'product':
                    temp_context += f'Product id: {temp_node} '
                else:
                    temp_context += f'Paper id: {temp_node} '
                if include_title:
                    title = text['title'][temp_node]
                    temp_context += f"Title: {title}\n"
                choice_list.append(temp_context)
            random.shuffle(choice_list)

            right_loc = 0
            for i in range(len(choice_list)):
                if choice_list[i].find(f'{one_hop_node}') != -1:
                    right_loc = i
                    break
            sign = ['A', 'B', 'C', 'D']
            question = '\n\nBased on the available partial information. Which other node can be a 2-hop neighbor of Target node1\n '
            for i in range(4):
                question += f'{sign[i]}. {choice_list[i]}'
            question += '\nDo not provide your reasoning. The answer should be A, B, C or D.\n Answer:\n\n'

            answer = sign[right_loc]

            json_data = {
                "Context": context,
                "Question": question,
                "Answer": answer
            }

            output_file = path + f'/{dir}/case8.jsonl'
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "a") as f:
                f.write(json.dumps(json_data) + "\n")
        else:
            print('case is wrong!')
    print(output_file + ' has completed!')





def get_right_wrong_middle_nodes(all_hops, target_node_2_index, edge_index):
    one_hop_nodes = all_hops[0]
    target_node_2_one_hop_nodes = get_subgraph(target_node_2_index, edge_index, hop=1)
    target_node_2_one_hop_nodes = target_node_2_one_hop_nodes[0]
    right_middle_nodes = list(set(target_node_2_one_hop_nodes) & set(one_hop_nodes))
    wrong_middle_nodes = list(set(one_hop_nodes) - set(target_node_2_one_hop_nodes))
    return right_middle_nodes, wrong_middle_nodes

def generate_structure_prompt_for_link(node_index, text, all_hops, data, max_papers_1,
                                                   max_papers_2, dataset, include_title, hop=1):
    np.random.seed(42)
    prompt_str = ""
    Target_word = "Product id: " if dataset == "product" else "Paper id: "

    for h in range(0, hop):
        neighbors_at_hop = all_hops[h]
        neighbors_at_hop = np.array(neighbors_at_hop)
        neighbors_at_hop = np.unique(neighbors_at_hop)
        np.random.shuffle(neighbors_at_hop)
        if h == 0:
            neighbors_at_hop = neighbors_at_hop[:max_papers_1]
        else:
            neighbors_at_hop = neighbors_at_hop[:max_papers_2]

        if len(neighbors_at_hop) > 0:
            if dataset != 'product':
                prompt_str += f"\n\nKnown neighbor papers at hop {h + 1} (partial, may be incomplete):\n"
            else:
                prompt_str += f"\n\nKnown neighbor products purchased toghther at hop {h + 1} (partial, may be incomplete):\n"

            for i, neighbor_idx in enumerate(neighbors_at_hop):
                prompt_str += f"\n{Target_word}{neighbor_idx}"
                if include_title:
                    neighbor_title = text['title'][neighbor_idx]
                    prompt_str += f"\nTitle: {neighbor_title}"
    return prompt_str

def sample_and_remove(all_hops):
    """
    Randomly samples one value from each list in all_hops and removes the sampled value.

    Args:
        all_hops (list of lists): A list where each sublist contains integers to sample from.

    Returns:
        tuple: A tuple containing:
            - sampled_values: List of randomly sampled values (one from each sublist).
            - updated_hops: The updated all_hops with sampled values removed.
    """
    sampled_values = []
    updated_hops = []

    for hop in all_hops:
        if hop:  # Ensure the list is not empty
            sampled_value = random.choice(hop)  # Randomly sample one value
            sampled_values.append(sampled_value)
            updated_hop = [x for x in hop if x != sampled_value]  # Remove the sampled value
            updated_hops.append(updated_hop)
        else:
            sampled_values.append(None)  # Handle empty lists gracefully
            updated_hops.append([])

    return sampled_values, updated_hops

def filter_train_or_test_nodes(data, all_hops, is_train):
    if not is_train:
        return all_hops
    mask = data.train_mask
    filtered_hops = []
    for hop_nodes in all_hops:
        if len(hop_nodes) == 0:
            filtered_hops.append([])
            continue
        # Convert hop_nodes to tensor if it's a list
        hop_nodes = torch.tensor(hop_nodes) if not isinstance(hop_nodes, torch.Tensor) else hop_nodes

        # Apply the mask to filter nodes
        filtered_hop = hop_nodes[mask[hop_nodes]]
        filtered_hops.append(filtered_hop.tolist())  # Convert back to list if needed

    return filtered_hops


def generate_structure_prompt(node_index, text, all_hops, data, hop, max_papers_1,
                              max_papers_2, abstract_len, include_label, dataset, is_train, mode):
    """
    Handle neighbors when attention is not used.

    Parameters:
        node_index (int): Index of the target node.
        text: Textual information of the node.
        all_hops (list): List of all neighbor nodes up to a certain hop.
        data: Graph data object.
        hop (int): Number of hops to consider.
        max_papers_1 (int): Maximum number of papers for the first hop.
        max_papers_2 (int): Maximum number of papers for the second hop.
        abstract_len (int): Length of the abstract to consider.
        include_label (bool): Whether to include labels.
        dataset (str): Name of the dataset being used.

    Returns:
        str: String containing information about standard neighbors.
    """
    np.random.seed(42)
    prompt_str = ""
    if dataset in ["product", "Amazon"]:
        Target_word = "Product id"
    elif dataset == 'Actor':
        Target_word = "Actor id"
    else:
        Target_word = "Paper id"

    for h in range(0, hop):
        neighbors_at_hop = all_hops[h]
        neighbors_at_hop = np.array(neighbors_at_hop)
        neighbors_at_hop = np.unique(neighbors_at_hop)
        np.random.shuffle(neighbors_at_hop)
        if h == 0:
            neighbors_at_hop = neighbors_at_hop[:max_papers_1]
        else:
            neighbors_at_hop = neighbors_at_hop[:max_papers_2]

        if len(neighbors_at_hop) > 0:
            if dataset == 'Actor':
                prompt_str += f"\nKnown co-occurred actors on the same Wikipedia page at hop {h + 1} (partial, may be incomplete):\n"
            elif dataset in ['product', 'Amazon']:
                prompt_str += f"\nKnown neighbor products purchased toghther at hop {h + 1} (partial, may be incomplete):\n"
            else:
                prompt_str += f"\nKnown neighbor papers at hop {h + 1} (partial, may be incomplete):\n"

            if mode != 'pure structure':
                if dataset in ['Actor', 'Amazon']:
                    for i, neighbor_idx in enumerate(neighbors_at_hop):
                        neighbor_content = text['content'][neighbor_idx]
                        prompt_str += f"\n{Target_word}{neighbor_idx}\nContent: {neighbor_content}\n"

                        if is_train:
                            if include_label and data.train_mask[neighbor_idx]:
                                label = text['label'][neighbor_idx]
                                prompt_str += f"Label: {label}\n"
                        else:
                            if include_label and (data.train_mask[neighbor_idx] or data.val_mask[neighbor_idx]):
                                label = text['label'][neighbor_idx]
                                prompt_str += f"Label: {label}\n"
                else:
                    for i, neighbor_idx in enumerate(neighbors_at_hop):
                        neighbor_title = text['title'][neighbor_idx]
                        prompt_str += f"\n{Target_word}{neighbor_idx}\nTitle: {neighbor_title}\n"

                        if abstract_len > 0:
                            neighbor_abstract = text['abs'][neighbor_idx]
                            prompt_str += f"Abstract: {neighbor_abstract[:abstract_len]}\n"

                        if is_train:
                            if include_label and data.train_mask[neighbor_idx] :
                                label = text['label'][neighbor_idx]
                                prompt_str += f"Label: {label}\n"
                        else:
                            if include_label and (data.train_mask[neighbor_idx] or data.val_mask[neighbor_idx]):
                                label = text['label'][neighbor_idx]
                                prompt_str += f"Label: {label}\n"
            else:
                for i, neighbor_idx in enumerate(neighbors_at_hop):
                    prompt_str += f"\n{Target_word}{neighbor_idx}"

                    if is_train:
                        if include_label and data.train_mask[neighbor_idx]:
                            label = text['label'][neighbor_idx]
                            prompt_str += f" Label: {label}\n"
                    else:
                        if include_label and (data.train_mask[neighbor_idx] or data.val_mask[neighbor_idx]):
                            label = text['label'][neighbor_idx]
                            prompt_str += f" Label: {label}\n"
    return prompt_str