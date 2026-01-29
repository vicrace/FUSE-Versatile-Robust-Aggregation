import argparse
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from transformers import logging
logging.set_verbosity_error()

folder_path = "....../1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled" # download 1bw dataset and replace the path

def evaluate_1bw(model_path, batch_size=8, stride=512, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()

    test_path = Path(folder_path)
    if not test_path.exists():
        raise ValueError(f"Test folder {folder_path} does not exist")

    all_lines = []
    for txt_file in sorted(test_path.glob("news.en.*")): 
        with open(txt_file, "r", encoding="utf-8") as f:
            all_lines.extend([line.strip() for line in f if line.strip()])

    print(f"Concatenating {len(all_lines)} sentences ...")
    text = "\n".join(all_lines)

    encodings = tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=False)
    seq_len = encodings.input_ids.size(1)

    max_length = model.config.n_positions
    stride = min(stride, max_length)
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
    print(f"Perplexity on 1BW heldout: {ppl.item():.4f}")
    return ppl.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    evaluate_1bw(args.model_path, args.batch_size)
