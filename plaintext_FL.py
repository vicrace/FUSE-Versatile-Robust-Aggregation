import copy
import os
from tqdm import tqdm
import numpy as np
import math
import torch
import time

from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, get_scheduler, DataCollatorForLanguageModeling,AutoConfig, set_seed
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training, PeftModel

from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args

evaluation = False
set_seed(42) #Set seed: 42, 99, 2023

# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)

####log console output
import sys
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")  # append mode so logs persist
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # make sure it writes in real-time
    def flush(self):
        self.terminal.flush()
        self.log.flush()

log_filename = os.path.join(script_args.output_dir, f"training_console_output.txt") 
sys.stdout = Logger(log_filename)
sys.stderr = sys.stdout  # also log stderr if needed
    
save_config(script_args, fed_args)
print(script_args, fed_args)

# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token   

# ===== Load the dataset =====
dataset = get_dataset(script_args.dataset_name, script_args.local_data_dir)
train_dataset, inference_dataset = process_sft_dataset(script_args.dataset_name, dataset, script_args.dataset_sample, script_args.inference_sample)
train_dataset = train_dataset["train"] 

# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

### Define weight from scratch
config = AutoConfig.from_pretrained(script_args.model_name_or_path)
model = AutoModelForCausalLM.from_config(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
   
# ===== Split the dataset into clients =====
local_datasets = split_dataset(fed_args, script_args, train_dataset)
local_infdatasets = split_dataset(fed_args, script_args, inference_dataset)
#local_testdatasets = split_dataset(fed_args, script_args, test_dataset)
sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]
print(sample_num_list)

if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

if training_args.gradient_checkpointing:
    model.enable_input_require_grads()

# ===== Define the global and local models =====
global_dict = copy.deepcopy(get_peft_model_state_dict(model))
local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
local_sparse_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

# ===== Define the formatting function (cater to TRL SFTTrainer)=====
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

formatting_prompts_func = None
response_template = None
response_template_ids = None

if "gpt2" not in script_args.model_name_or_path.lower():
    formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:] 


# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]

for round_idx in tqdm(range(fed_args.num_rounds)):

    clients_this_round = get_clients_this_round(fed_args, round_idx)

    print(f">> ==================== Round {round_idx+1} : {clients_this_round} ====================")
    
    for client in range(fed_args.num_clients):

        if client not in clients_this_round:
            training_loss[client].append(-1)            # -1 is an indicator of not training
            continue

        set_peft_model_state_dict(model, global_dict)   # sync the global model to the local model

        sub_dataset = get_dataset_this_round(local_datasets[client], round_idx, fed_args, script_args)
        sub_infset  = get_dataset_this_round(local_infdatasets[client], round_idx, fed_args, script_args)

        new_lr = cosine_learning_rate(round_idx, fed_args.num_rounds, script_args.learning_rate, 1e-6)      # manually schedule the learning rate
        training_args = get_training_args(script_args, new_lr)

        ##===== Train local model on the client side =====
        trainer = get_fed_local_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            local_dataset=sub_dataset,
            val_dataset=sub_infset,
            formatting_prompts_func=formatting_prompts_func,
            data_collator=data_collator,
            global_dict=global_dict,
            fed_args=fed_args,
            script_args=script_args,
            local_auxiliary=auxiliary_model_list[client],
            global_auxiliary=global_auxiliary,
        )
        results = trainer.train()
        training_loss[client].append(results.training_loss)
        
        if evaluation:
            eval_results = trainer.evaluate() 
            eval_loss = eval_results.get("eval_loss", eval_results.get("loss", None))
            if eval_loss is not None:
                perplexity = math.exp(eval_loss) if not math.isinf(eval_loss) else float("inf")
            else:
                perplexity = None

            print(f"Client {client} Eval — loss: {eval_loss}, ppl: {perplexity}")


        # ===== Client transmits local information to server =====
        local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!

    # ===== Server aggregates the local models =====
    start_time = time.perf_counter()
    global_dict, global_auxiliary = global_aggregate(
        fed_args, global_dict, local_dict_list, sample_num_list, \
        clients_this_round, round_idx, proxy_dict=proxy_dict, \
        opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict)
    )
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"Plaintext FL Execution time: {elapsed_time} seconds")
    set_peft_model_state_dict(model, global_dict)   # Update global model

    # ===== Save the model =====
    if (round_idx+1) % fed_args.save_model_freq == 0:
        trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round_idx+1}"))
    
    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))

