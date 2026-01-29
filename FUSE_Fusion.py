import copy
import os
from tqdm import tqdm
import numpy as np
import math
import torch
import time
import builtins
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer, logging, DataCollatorForLanguageModeling,AutoConfig, set_seed
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training, PeftModel

from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args

from openfhe import *
import os, psutil

evaluation = False

# ===== Define the sparsification N binary combine slot
n_combine = 8   #for quartile-ranking - define combine how many mask together as 1 slot
quartile = 4    #for quartile-ranking - define how many quartile needed

set_seed(42)

# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)

#FUSE-SPA
D=10
l=16
ISlot = math.ceil(math.log(fed_args.sample_clients, D)) + 1
n_combine = int(math.floor(l/ISlot))

#FUSE-RDX
scale = 10**script_args.rdx_scale
MAX_ABS = 5
A = MAX_ABS * scale 
Wtotal = 10
WIDTH = 2*Wtotal*A

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
sys.stderr = sys.stdout 

save_config(script_args, fed_args)
print(script_args, fed_args)

# ===== Load the dataset =====
dataset = get_dataset(script_args.dataset_name, script_args.local_data_dir)
train_dataset, inference_dataset = process_sft_dataset(script_args.dataset_name, dataset, script_args.dataset_sample, script_args.inference_sample)
train_dataset = train_dataset["train"]

temp_time = 0.0
client_time =0.0
server_time = 0.0
# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

config = AutoConfig.from_pretrained(script_args.model_name_or_path)
model = AutoModelForCausalLM.from_config(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.config.use_cache = False  

if training_args.gradient_checkpointing:
    model.enable_input_require_grads()


# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token   

# ===== Split the dataset into clients =====
local_datasets = split_dataset(fed_args, script_args, train_dataset)
local_infdatasets = split_dataset(fed_args, script_args, inference_dataset)
sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]

# ===== Define the global and local models =====
global_dict = copy.deepcopy(get_peft_model_state_dict(model))
local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
local_sparse_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
local_radix_list = [[] for i in range(fed_args.num_clients)]
sparseK_index = [[] for _ in range(fed_args.num_clients)]
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

def topk_sparsification(state_dict, k):
    grads = []
    shapes = []

    for key, tensor in state_dict.items():
        grads.append(tensor.view(-1))
        shapes.append(tensor.shape)
    total_elements = sum([torch.tensor(shape).prod().item() for shape in shapes])
    true_k = math.floor(k/100 * total_elements)
    flat_grads = torch.cat(grads)
    topk_values, topk_indices = torch.topk(flat_grads.abs(), true_k)
    topk_values = flat_grads[topk_indices]

    #============ Convert index to binary slot with quartile range===========
    global temp_time
    start_time = time.perf_counter()
    min_value = topk_values.min().item()
    max_value = topk_values.max().item()
    range_value = max_value - min_value

    quantiles = torch.linspace(min_value, max_value, script_args.spa_bin+1)

    quantile_indices = torch.bucketize(topk_values, quantiles, right=False)
    quantile_indices = quantile_indices.clamp(min=1)
  
    global_mask = torch.zeros(total_elements, dtype=torch.uint8)
    quantile_indices = quantile_indices.to(torch.uint8)
    global_mask[topk_indices] = quantile_indices
   
    sparse_index_length = math.ceil(len(global_mask)/n_combine)
    global_mask = list(map(int, global_mask))

    remainder = len(global_mask) % n_combine
    if remainder != 0:
        pad_length = n_combine - remainder
        global_mask += [0] * pad_length

    grouped_mask = []

    for i in range(0, len(global_mask), n_combine):
        group = global_mask[i:i + n_combine]
        padded_group = [f"{num:0{2}d}" for num in group]
        group_str = ''.join(padded_group)
        grouped_mask.append(int(group_str))
    
    end_time = time.perf_counter()
    temp_time += end_time - start_time
    temp_time /= fed_args.sample_clients

    return topk_values, topk_indices, grouped_mask, shapes

def reconstruct_gradients_sparse(state_dict, topk_indices, total_param):

    grads = []
    shapes = []

    for key, tensor in state_dict.items():
        grads.append(tensor.view(-1))
        shapes.append(tensor.shape)

    flat_grads = torch.cat(grads)
    selected_gradients = torch.zeros_like(flat_grads)
    selected_gradients[topk_indices] = flat_grads[topk_indices] 
    new_state_dict = {}
    pointer = 0
    for (key, tensor), shape in zip(state_dict.items(), shapes):
        num_elements = torch.tensor(shape).prod().item()
        new_state_dict[key] = selected_gradients[pointer:pointer + num_elements].view(shape)
        pointer += num_elements
  
    return new_state_dict

def reconstruct_fullgrad_dict(state_dict, agg_gradients, topk_indices, shapes, total_params):

    grads = []
    for key, tensor in state_dict.items():
        grads.append(tensor.view(-1))
    flat_grads = torch.cat(grads)
    flat_grads[topk_indices] = agg_gradients 

    new_state_dict = {}
    pointer = 0
    for (key, tensor), shape in zip(state_dict.items(), shapes):
        num_elements = torch.tensor(shape).prod().item()
        new_state_dict[key] = flat_grads[pointer:pointer + num_elements].view(shape)
        pointer += num_elements
  
    return new_state_dict

def encrypt_gradient_for_aggregation(local_list,indexes):
    grads = []
    shapes = []

    for key, tensor in local_list.items():
        grads.append(tensor.view(-1))
        shapes.append(tensor.shape)
    flat_grads = torch.cat(grads)
    selected_gradients = flat_grads[indexes]
    
    _ ,ciphertexts = encrypt_pair__cipher(selected_gradients)
    return ciphertexts

def encrypt_vector_to_ciphertext(indices): #now the len(indices) is 196608, then max slot is 32768 which consume 6 ciphertexts.
    count = 0
    ciphertexts = []

    for i in range(0, len(indices), max_slots):
        chunk = indices[i:i+max_slots]
        pt = cc.MakeCKKSPackedPlaintext(chunk)
        ct = cc.Encrypt(keys.publicKey, pt)
        ciphertexts.append(ct)
        count=count+1
    return ciphertexts

def encrypt_full_gradient(local_dict_list):
    flattened_tensors = [param.flatten() for param in local_dict_list.values()]
    full_vector = torch.cat(flattened_tensors, dim=0).tolist()  
    count = 0
    
    ciphertexts = []
    for i in range(0, len(full_vector), max_slots):
        chunk = full_vector[i:i+max_slots]
        pt = cc.MakeCKKSPackedPlaintext(chunk)
        ct = cc.Encrypt(keys.publicKey, pt)
        ciphertexts.append(ct)
        count=count+1
    return ciphertexts

def decode_index_packing(value, digits_per_slot=2):
    clean_val = builtins.round(val.real) if isinstance(val, complex) else round(val)
    total_digits = n_combine * digits_per_slot
    val_str = str(clean_val).zfill(total_digits)
    slots = [int(val_str[i:i+digits_per_slot]) for i in range(0, total_digits, digits_per_slot)]

    return slots

def encode_pair_signed(a, b, scale, width):
    a_int = int(round(a * scale))
    b_int = int(round(b * scale))

    return a_int * width + b_int

def decode_pair_signed(z, width, scale):
    a_int = z // width
    b_int = z % width
  
    half = width // 2
    if b_int >= half:
        b_int -= width
    if a_int >= half:
        a_int -= width
    return a_int / scale, b_int / scale

def encrypt_pair__cipher(datas):
    global temp_time
    start_time = time.perf_counter()
    flattern_vector = datas.tolist()
    
    full_vector = []
    i = 0
    while i < len(flattern_vector) - 1: 
        a, b = flattern_vector[i], flattern_vector[i + 1] 
        z = encode_pair_signed(a, b, scale, WIDTH)
        full_vector.append(z)
        i += 2
   
    lenght = len(full_vector)
    count = 0
    ciphertexts = []
    for i in range(0, len(full_vector), max_slots):
        chunk = full_vector[i:i+max_slots]
        pt = cc.MakeCKKSPackedPlaintext(chunk)
        ct = cc.Encrypt(keys.publicKey, pt)
        ciphertexts.append(ct)
        count=count+1

    end_time = time.perf_counter()
    temp_time += (end_time - start_time)
    return lenght, ciphertexts

# ===== OpenFHE setup ====
mult_depth = 2 
scale_mod_size = 50 
descale = Wtotal

if fed_args.fed_alg == "fednormbound":
    scale_mod_size = 55
    mult_depth=21
    descale = Wtotal*Wtotal


ringdim = 65536
max_slots = int(ringdim / 2)

parameters = CCParamsCKKSRNS()
parameters.SetRingDim(ringdim)
parameters.SetMultiplicativeDepth(mult_depth)
parameters.SetScalingModSize(scale_mod_size)

cc = GenCryptoContext(parameters)
cc.Enable(PKESchemeFeature.PKE)
cc.Enable(PKESchemeFeature.KEYSWITCH)
cc.Enable(PKESchemeFeature.LEVELEDSHE)

total_param = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)

num_ciphertexts = math.ceil((total_param * script_args.topk / 100) / float(cc.GetRingDimension() // 2)) 
num_ciphertexts = math.ceil(num_ciphertexts / 2)
print("Max Ciphertext Need " + str(num_ciphertexts))

keys = cc.KeyGen()
cc.EvalMultKeyGen(keys.secretKey)
encrypted_dict_list = [[] for _ in range(fed_args.num_clients)]
encrypted_index_list = [[] for _ in range(fed_args.num_clients)]

zero_vector = [0.0] * (cc.GetRingDimension() // 2)
zero_plaintext = cc.MakeCKKSPackedPlaintext(zero_vector)
global_encrypted_list = [cc.Encrypt(keys.publicKey, zero_plaintext) for _ in range(num_ciphertexts)]
l2_norm = [cc.Encrypt(keys.publicKey, zero_plaintext) for _ in range(fed_args.num_clients)]

# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]

print("Method - hybrid - sparse-k" ,  script_args.topk)
for round_num in tqdm(range(fed_args.num_rounds)):
    client_time = 0.0
    server_time =0.0
    clients_this_round = get_clients_this_round(fed_args, round_num)

    print(f">> ==================== Round {round_num+1} : {clients_this_round} ====================")
    index_len = 0
    sparsek = 0
    pshape = []
    for client in range(fed_args.num_clients):

        if client not in clients_this_round:
            training_loss[client].append(-1)          
            continue

        set_peft_model_state_dict(model, global_dict)   # sync the global model to the local model

        sub_dataset = get_dataset_this_round(local_datasets[client], round_num, fed_args, script_args)    
        sub_infset = get_dataset_this_round(local_infdatasets[client], round_num, fed_args, script_args)  
        new_lr = cosine_learning_rate(round_num, fed_args.num_rounds, script_args.learning_rate, 1e-6)     
        training_args = get_training_args(script_args, new_lr)

        # ##===== Train local model on the client side =====
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
            perplexity = np.exp(eval_results['eval_loss'])
            print("Evaluation Results:", eval_results, "    perplexity: ", perplexity)

        # ===== Client transmits local information to server =====
        local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!

        if fed_args.fed_alg == 'fednormbound':
            flattern = [local_dict_list[client][key].view(-1) for key in global_dict.keys()]
            flat_values = torch.cat(flattern)

            update_norm = torch.norm(flat_values, p=2) ##find l2 norm
            norm_vector = [1/update_norm] * (cc.GetRingDimension() // 2)

            pt = cc.MakeCKKSPackedPlaintext(norm_vector)
            ct = cc.Encrypt(keys.publicKey, pt)
            l2_norm[client] = ct

        #### FUSE-SPA
        local_sparse_list[client] = copy.deepcopy(get_peft_model_state_dict(model))
        topk_values , top_indices, sparseK_index[client], pshape = topk_sparsification(local_sparse_list[client], k=script_args.topk)
        sparsek = len(topk_values)

        start_time = time.perf_counter()
        ciphertext = encrypt_vector_to_ciphertext(sparseK_index[client])
        end_time = time.perf_counter()
        temp_time += (end_time - start_time)
        index_len = len(ciphertext)
        encrypted_index_list[client] = []
        encrypted_index_list[client] = ciphertext

        for j in range(index_len):  
            filename = f"ciphertexts/index/ciphertext_{client*index_len+j}.txt"
            if not SerializeToFile(filename, encrypted_index_list[client][j], BINARY):      
                print(f"Error writing ciphertext {client*index_len+j}")
                continue

    temp_time /= fed_args.sample_clients
    client_time += temp_time

    # ===== Exchange index =====
    
    zero_vector = [0] * (cc.GetRingDimension() // 2)
    zero_plaintext = cc.MakeCKKSPackedPlaintext(zero_vector) ##if want to use this ciphertext add with another then must no use [None] * 36, because this is null
    global_index_list = [cc.Encrypt(keys.publicKey, zero_plaintext) for _ in range(index_len)]

    start_time = time.perf_counter()
    encrypted_glpbal_index = exchange_indices(clients_this_round, sparseK_index, encrypted_index_list, global_index_list, cc)
    end_time = time.perf_counter()
    server_time += (end_time - start_time)

    start_time = time.perf_counter()
    final_index = []
    for index in range(len(encrypted_glpbal_index)):
        decrypted_ciphertext = cc.Decrypt(encrypted_glpbal_index[index], keys.secretKey)
        values = decrypted_ciphertext.GetCKKSPackedValue()
        final_index.extend(values)

    final_index = final_index[:]
    decoded_index = []
    for val in final_index:
        decoded = decode_index_packing(val)
        decoded_index.extend(decoded) 

    decoded_quartile = torch.tensor(decoded_index[:total_param]) 
    sorted_vals, sorted_idx = decoded_quartile.sort(descending=True) 
    selected_indices, _ = sorted_idx[:sparsek].sort(descending=False)
    end_time = time.perf_counter()
    client_time +=(end_time - start_time)

    for client in range(fed_args.num_clients):
        if client not in clients_this_round:
            continue
        local_sparse_list[client] = reconstruct_gradients_sparse(local_sparse_list[client], selected_indices, total_param)

        start_time = time.perf_counter()
        grads = []
        for key, tensor in local_dict_list[client].items():
            grads.append(tensor.view(-1))

        flat_grads = torch.cat(grads)
        extracted_values = flat_grads[selected_indices]

        ##### FUSE-RDX
        ciphertext = encrypt_gradient_for_aggregation(local_dict_list[client],selected_indices) 
        end_time = time.perf_counter()
        encrypted_dict_list[client] = []
        encrypted_dict_list[client] = ciphertext
        temp_time += (end_time - start_time)
        
        for j in range(len(ciphertext)):  
            filename = f"ciphertexts/updates/ciphertext_{client*len(ciphertext)+j}.txt"
            if not SerializeToFile(filename, encrypted_dict_list[client][j], BINARY):     
                print(f"Error writing ciphertext {client*len(ciphertext)+j}")
                continue
    temp_time /= fed_args.sample_clients
    client_time += temp_time

    # ===== Server aggregates the local models =====
    start_time = time.perf_counter()
    encrypted_aggregates, global_auxiliary = global_aggregate_encrypted_pairing(
        fed_args, global_dict, sample_num_list, \
        clients_this_round, round_num, global_encrypted_list, encrypted_dict_list, cc,keys.publicKey, sparsek,proxy_dict=proxy_dict, \
        opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict,l2_norm) 
    )
    end_time = time.perf_counter()
    server_time += (end_time - start_time)

    computed_values = []
    for index in range(len(encrypted_aggregates)):
        decrypted_ciphertext = cc.Decrypt(encrypted_aggregates[index], keys.secretKey)
        values = decrypted_ciphertext.GetCKKSPackedValue()
        computed_values.extend(values)
   
    start_time = time.perf_counter()
    resut_val = []
    for val in computed_values:
        decoded = val.real
        resut_val.append(decoded)


    computed_gradients = []
    for z in resut_val:
        a_rec, b_rec = decode_pair_signed(z, width=WIDTH, scale=scale) 
        computed_gradients.extend([a_rec /descale, b_rec /descale])
    end_time = time.perf_counter()
    client_time += (end_time - start_time)

    computed_gradients = torch.tensor(computed_gradients, dtype=torch.float32)
    global_dict = reconstruct_fullgrad_dict(global_dict, computed_gradients[0:sparsek], selected_indices, pshape, total_param) 
    set_peft_model_state_dict(model, global_dict)  

    print(f"FUSE-Fusion Time: server time {server_time}s , client time {client_time}s")

    zero_vector = [0.0] * (cc.GetRingDimension() // 2)
    zero_plaintext = cc.MakeCKKSPackedPlaintext(zero_vector) 
    global_encrypted_list = [cc.Encrypt(keys.publicKey, zero_plaintext) for _ in range(num_ciphertexts)]

    ##reset l2_norm
    if fed_args.fed_alg == 'fednormbound':
        l2_norm = [cc.Encrypt(keys.publicKey, zero_plaintext) for _ in range(fed_args.num_clients)]

    # ===== Save the model =====
    if (round_num+1) % fed_args.save_model_freq == 0:
        trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round_num+1}"))
    
    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))
