import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig
from datasets import load_dataset
from torch.utils.data import DataLoader

def evaluate_perplexity(model_path, batch_size=8, num_proc=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()

    dataset = load_dataset("cimec/lambada", split="train")
    print(f"Loaded {len(dataset)} samples from LAMBADA.")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=num_proc)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, pin_memory=True)

    total_loss = 0
    total_tokens = 0

    with torch.inference_mode():
        for batch in dataloader:
            # Move to GPU
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            # Count tokens (exclude padding)
            non_pad_tokens = attention_mask.sum().item()
            total_loss += loss.item() * non_pad_tokens
            total_tokens += non_pad_tokens

    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    print(f"Perplexity on LAMBADA: {perplexity.item():.4f}")
    return perplexity.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="openai-community/gpt2")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_proc", type=int, default=32, help="Parallel workers for tokenization")
    args = parser.parse_args()

    evaluate_perplexity(args.model_path, args.batch_size, args.num_proc)
