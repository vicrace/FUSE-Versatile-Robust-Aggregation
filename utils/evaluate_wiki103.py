import argparse
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig
from datasets import load_dataset
from transformers import logging
logging.set_verbosity_error()


def evaluate_wikitext(model_path, batch_size=8, num_proc=4):
    dataset_name = "Salesforce/wikitext"
    subset_name = "wikitext-103-raw-v1"
    split_name = "test"
    stride = 512

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()

    print(f"Loading dataset {dataset_name} ({subset_name}) split={split_name}")
    dataset = load_dataset(dataset_name, subset_name, split=split_name, num_proc=num_proc)

    print("Concatenating test data ...")
    text = "\n\n".join(dataset["text"])

    encodings = tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=False)
    seq_len = encodings.input_ids.size(1)

    max_length = model.config.n_positions
    stride = 512 
    nlls = []
    total_tokens = 0

    for i in range(0, seq_len - 1, stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, seq_len)
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-stride] = -100 

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * (end_loc - begin_loc)

        nlls.append(neg_log_likelihood)
        total_tokens += end_loc - begin_loc

    ppl = torch.exp(torch.stack(nlls).sum() / total_tokens)
    print(f"Perplexity on WikiText-103-v1 (test split): {ppl.item():.4f}")
    return ppl.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_proc", type=int, default=4, help="Parallel workers for dataset loading")
    args = parser.parse_args()

    evaluate_wikitext(args.model_path, args.batch_size, args.num_proc)
