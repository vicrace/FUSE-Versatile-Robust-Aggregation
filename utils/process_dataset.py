import datasets
from datasets import load_dataset, concatenate_datasets
import pandas as pd
from .conversation import get_conv_template
from functools import partial

def get_dataset(dataset_name, local_data_dir=None):

    if dataset_name in ["sytelus/openwebtext"]:
        dataset = load_dataset(dataset_name)
    else:
        dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        dataset = load_dataset(dataset_name, split="train")
    return dataset

def process_sft_dataset(dataset_name, dataset, dataset_sample, inference_sample):
    if dataset_name in ["sytelus/openwebtext"]:
        dataset = dataset["train"].select(range(dataset_sample))
        trainset = dataset.train_test_split(test_size=0.3, seed=42) ## 42,99

        inference = trainset["test"]
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")
    
    dataset = dataset.shuffle(seed=42) #42,99
    if dataset_sample and dataset_name not in ['sytelus/openwebtext']:
        num_total = len(dataset)      
        num_train = min(num_total,dataset_sample)

        trainset = dataset.select(range(num_train))
        inference = dataset.select(range(num_train,num_train+inference_sample)) 
    return trainset, inference

def alpaca_format(example):
    if example['input'] == "":
        example["instruction"] = example["instruction"]
    else:
        example["instruction"] = example["instruction"] + " " + example['input']
    example["response"] = example['output']
    return example


def process_dpo_dataset(dataset_name, dataset, template_name, dataset_sample):
    if dataset_name in ["Anthropic/hh-rlhf"]:
        dataset = dataset.map(partial(split_hh, template_name=template_name), load_from_cache_file=False)
    elif dataset_name in ["HuggingFaceH4/ultrafeedback_binarized"]:
        dataset = dataset.map(partial(split_ultrafeedback, template_name=template_name), load_from_cache_file=False)
        dataset = dataset.remove_columns(['prompt_id', 'messages', 'score_chosen', 'score_rejected'])
    
    dataset = dataset.shuffle(seed=2023)
    if dataset_sample:
        num_sample = min(len(dataset), dataset_sample)
        dataset = dataset.select(range(num_sample))
    print(f">> ===== After processing, Dataset {dataset_name} has {len(dataset)} examples. =====")
    print(f">> ===== Data Example =====")
    print(dataset[0])
    print(f">> {'='*50}")
    return dataset
    
def find_common_prefix(str1, str2):
    prefix = ""
    for i in range(min(len(str1), len(str2))):
        if str1[i] == str2[i]:
            prefix += str1[i]
        else:
            break
    return prefix

def split_ultrafeedback(example, template_name="vicuna_v1.1"):
    conv_template = get_conv_template(template_name)

    conv_template.append_message(conv_template.roles[0], example["prompt"])
    conv_template.append_message(conv_template.roles[1], None)
    example["prompt"] = conv_template.get_prompt()
    example["chosen"] = " " + example["chosen"][1]["content"]       # There might need a space in the front.
    example["rejected"] = " " + example["rejected"][1]["content"]
    return example

def split_hh(example, template_name="vicuna_v1.1"):
    common_prefix = find_common_prefix(example["chosen"], example["rejected"])

    conv_template = get_conv_template(template_name)

    sentence = common_prefix
    human_prefix_len = len("\n\nHuman: ")
    assistant_prefix_len = len("\n\nAssistant: ")
    sentence = sentence[human_prefix_len:]
    turn = "user"
    while True:
        if turn == "user":
            index = sentence.find("\n\nAssistant: ")
            if index == -1:
                break
            else:
                conv_template.append_message(conv_template.roles[0], sentence[:index])
                turn = "assistant"
                sentence = sentence[index + assistant_prefix_len :]
        elif turn == "assistant":
            index = sentence.find("\n\nHuman: ")
            if index == -1:
                break
            else:
                conv_template.append_message(conv_template.roles[1], sentence[:index])
                turn = "user"
                sentence = sentence[index + human_prefix_len :]
    conv_template.append_message(conv_template.roles[1], None)
    example["prompt"] = conv_template.get_prompt()
    example["chosen"] = example["chosen"][len(common_prefix) - 1 :]     # -1 to include the space in the front.
    example["rejected"] = example["rejected"][len(common_prefix) - 1 :]
    return example
