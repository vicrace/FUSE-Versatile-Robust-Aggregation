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
set_seed(42)  #42,99

# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)

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
sys.stderr = sys.stdout  # also log stderr if needed

save_config(script_args, fed_args)
print(script_args, fed_args)

# ===== Load the dataset =====
dataset = get_dataset(script_args.dataset_name, script_args.local_data_dir)
train_dataset, inference_dataset = process_sft_dataset(script_args.dataset_name, dataset, script_args.dataset_sample, script_args.inference_sample)
train_dataset = train_dataset["train"]

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

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

if training_args.gradient_checkpointing:
    model.enable_input_require_grads()

# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token   

# ===== Split the dataset into clients =====
#local_datasets = split_dataset(fed_args, script_args, dataset)
local_datasets = split_dataset(fed_args, script_args, train_dataset)
local_infdatasets = split_dataset(fed_args, script_args, inference_dataset)
sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]

# ===== Define the global and local models =====
global_dict = copy.deepcopy(get_peft_model_state_dict(model))
local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
local_sparse_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)
client_time =0.0
server_time =0.0
temp_time =0.0

# ===== Define the formatting function (cater to TRL SFTTrainer)=====
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

formatting_prompts_func = None
response_template = None
response_template_ids = None

if "gpt2" not in script_args.model_name_or_path.lower():
    formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:] 


def reconstruct_gradient_full_no_sparse(global_dict,resut_val,inplace= False):
    computed_gradients = []
    
    for z in resut_val:
        a_rec, b_rec = decode_pair_signed(z, width=WIDTH, scale=scale) 
        computed_gradients.extend([a_rec /descale, b_rec /descale]) 
    computed_gradients = torch.tensor(computed_gradients, dtype=torch.float32)

    
    if isinstance(computed_gradients, np.ndarray):
        flat = torch.from_numpy(computed_gradients)
    elif torch.is_tensor(computed_gradients):
        flat = computed_gradients.detach().cpu()
    else:
        flat = torch.tensor(computed_gradients)

    flat = flat.view(-1)
    total_expected = 0
    for v in global_dict.values():
        if torch.is_tensor(v):
            total_expected += v.numel()
        else:
            total_expected += torch.tensor(v).numel()

    if flat.numel() != total_expected:
        raise ValueError(f"computed_gradients length {flat.numel()} != expected {total_expected}")

    new_state = {}
    ptr = 0
    for name, tmpl in global_dict.items():
        if torch.is_tensor(tmpl):
            tmpl_tensor = tmpl
        else:
            tmpl_tensor = torch.tensor(tmpl)

        numel = tmpl_tensor.numel()
        shape = tuple(tmpl_tensor.shape)
        seg = flat[ptr: ptr + numel].view(shape)
        try:
            orig_dtype = tmpl_tensor.dtype
        except Exception:
            orig_dtype = torch.float32
        seg = seg.to(orig_dtype)

        device = getattr(tmpl_tensor, "device", torch.device("cpu"))
        seg = seg.to(device)

        if inplace and torch.is_tensor(tmpl) and tmpl.shape == seg.shape and tmpl.device == seg.device and tmpl.dtype == seg.dtype:
            tmpl.data.copy_(seg)
            new_state[name] = tmpl
        else:
            new_state[name] = seg.clone()

        ptr += numel

    assert ptr == total_expected, "reconstruction pointer mismatch"
    return new_state

def encode_pair_signed(a, b, scale, width):
    a_int = int(round(a * scale))
    b_int = int(round(b * scale))
    return a_int * width + b_int #, width, scale

def decode_pair_signed(z, width, scale):
    a_int = z // width
    b_int = z % width
  
    half = width // 2
    if b_int >= half:
        b_int -= width
    if a_int >= half:
        a_int -= width
    return a_int / scale, b_int / scale

def encrypt_pair__cipher(local_dict_list):

    flattened_tensors = [param.flatten() for param in local_dict_list.values()]
    flattern_vector = torch.cat(flattened_tensors, dim=0).tolist() 
    
    full_vector = []
    i = 0
    while i < len(flattern_vector) - 1:  # Pairwise
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

    return lenght, ciphertexts


# ===== OpenFHE setup ====
mult_depth = 2   
scale_mod_size = 50
descale = Wtotal

if fed_args.fed_alg == "fednormbound":
    mult_depth=21
    scale_mod_size = 55 
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
cc.Enable(ADVANCEDSHE)

total_param = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)

num_ciphertexts = math.ceil((total_param * 50 / 100) / float(cc.GetRingDimension() // 2)) 
print("Max Ciphertext Need " + str(num_ciphertexts))

keys = cc.KeyGen()
cc.EvalMultKeyGen(keys.secretKey)
encrypted_dict_list = [[] for _ in range(fed_args.num_clients)]
encrypted_index_list = [[] for _ in range(fed_args.num_clients)]

zero_vector = [0.0] * (cc.GetRingDimension() // 2)
zero_plaintext = cc.MakeCKKSPackedPlaintext(zero_vector) ##if want to use this ciphertext add with another then must no use [None] * 36, because this is null
global_encrypted_list = [cc.Encrypt(keys.publicKey, zero_plaintext) for _ in range(num_ciphertexts)]
l2_norm = [cc.Encrypt(keys.publicKey, zero_plaintext) for _ in range(fed_args.num_clients)]

# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]

for round_num in tqdm(range(fed_args.num_rounds)):
    client_time = 0.0
    server_time =0.0
    clients_this_round = get_clients_this_round(fed_args, round_num)

    print(f">> ==================== Round {round_num+1} : {clients_this_round} ====================")
    lenght = 0
    for client in range(fed_args.num_clients):

        if client not in clients_this_round:
            training_loss[client].append(-1)            # -1 is an indicator of not training
            continue

        set_peft_model_state_dict(model, global_dict)   # sync the global model to the local model

        sub_dataset = get_dataset_this_round(local_datasets[client], round_num, fed_args, script_args)      
        sub_infset = get_dataset_this_round(local_infdatasets[client], round_num, fed_args, script_args)  
        new_lr = cosine_learning_rate(round_num, fed_args.num_rounds, script_args.learning_rate, 1e-6)    
        training_args = get_training_args(script_args, new_lr)

        #===== Train local model on the client side =====
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

            update_norm = torch.norm(flat_values, p=2)
            norm_vector = [1/update_norm] * (cc.GetRingDimension() // 2)

            pt = cc.MakeCKKSPackedPlaintext(norm_vector)
            ct = cc.Encrypt(keys.publicKey, pt)
            l2_norm[client] = ct

        start_time = time.perf_counter()
        lenght ,ciphertext = encrypt_pair__cipher(local_dict_list[client])
        encrypted_dict_list[client] = ciphertext
        end_time = time.perf_counter()
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
        clients_this_round, round_num, global_encrypted_list, encrypted_dict_list, cc,keys.publicKey, lenght,proxy_dict=proxy_dict, \
        opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict,l2_norm)
    )
    end_time = time.perf_counter()
    server_time += (end_time - start_time)

    start_time = time.perf_counter()
    computed_values = []
    for index in range(len(encrypted_aggregates)):
        decrypted_ciphertext = cc.Decrypt(encrypted_aggregates[index], keys.secretKey)
        values = decrypted_ciphertext.GetCKKSPackedValue()
        computed_values.extend(values)
   
    resut_val = []
    for val in computed_values:
        decoded = val.real 
        resut_val.append(decoded)

    reconstruct_gradient_full_no_sparse(global_dict, resut_val, inplace=True)
    end_time = time.perf_counter()
    client_time += (end_time - start_time)

    print(f"FUSE-RDX Time: server time {server_time}s , client time {client_time}s")

    set_peft_model_state_dict(model, global_dict)   
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
