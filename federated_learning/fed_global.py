import random
import torch
import math
import time
import copy
import os, psutil
#from openfhe import *
#from concurrent.futures import ThreadPoolExecutor, as_completed
from math import factorial
from typing import List
from openfhe import *

def mult_polynomial(p: List[float], q: List[float]) -> List[float]:
    if not p or not q:
        return []
    res = [0.0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a == 0.0:
            continue
        for j, b in enumerate(q):
            res[i + j] += a * b
    return res

def power_polynomial(poly: List[float], exp: int) -> List[float]:
    if exp < 0:
        raise ValueError("Negative exponent not supported for polynomials.")
    result = [1.0]
    base = poly[:]
    e = exp
    while e:
        if e & 1:
            result = mult_polynomial(result, base)
        e >>= 1
        if e:
            base = mult_polynomial(base, base)
    return result
    
def computeF(n):
    coeff = [0.0] * (2 * n + 2)
    x = [0,1]
    x2 = [1,0,-1]

    for i in range(0, n + 1):
        scalar = 1/pow(4.0, i) * (factorial(2*i)/(factorial(i)*factorial(i)))
        px2i = power_polynomial(x2, i)     # x2^i
        c = mult_polynomial(x, px2i)       # x * (x2^i)
        for j, cj in enumerate(c):
            coeff[j] += cj * scalar
           
    return coeff

def trimmed_linear(ct, cc):
    # f(x) = 0.5 * (x + 1)
    ct_add = cc.EvalAdd(ct, 1) 
    ct_half = cc.EvalMult(ct_add, 0.5)     
    return ct_half

def evaluate_comp(cc, poly, x):
    a0 = cc.MakeCKKSPackedPlaintext([poly[0]]*(cc.GetRingDimension()//2))
    a1 = cc.MakeCKKSPackedPlaintext([poly[1]]*(cc.GetRingDimension()//2))
    a2 = cc.MakeCKKSPackedPlaintext([poly[2]]*(cc.GetRingDimension()//2))
    a3 = cc.MakeCKKSPackedPlaintext([poly[3]]*(cc.GetRingDimension()//2))

    term = cc.EvalMult(x, a3)
    cc.RelinearizeInPlace(term)

    term = cc.EvalAdd(term, a2)
    term = cc.EvalMult(x, term)
    cc.RelinearizeInPlace(term)

    term = cc.EvalAdd(term, a1)
    term = cc.EvalMult(x, term)
    cc.RelinearizeInPlace(term)
    cc.RescaleInPlace(term)

    ct = cc.EvalAdd(term, a0)
    return ct

def get_clients_this_round(fed_args, round):
    if (fed_args.fed_alg).startswith('local'):
        clients_this_round = [int((fed_args.fed_alg)[-1])]
    else:
        if fed_args.num_clients < fed_args.sample_clients:
            clients_this_round = list(range(fed_args.num_clients))
        else:
            random.seed(round)
            clients_this_round = sorted(random.sample(range(fed_args.num_clients), fed_args.sample_clients))
    return clients_this_round

def trimmed_bound(ct, cc):
    # simple approximation: f(x) = 0.5 * (x + x^2 / (abs_max))
    ct_sq = cc.EvalMult(ct, ct)     
    cc.RelinearizeInPlace(ct_sq)

    ct_relu = cc.EvalAdd(ct, ct_sq)    
    ct_relu = cc.EvalMult(ct_relu, 0.5)     
    cc.RelinearizeInPlace(ct_relu)
    return ct_relu

def global_aggregate(fed_args, global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, proxy_dict=None, opt_proxy_dict=None, auxiliary_info=None):
    sample_this_round = sum([sample_num_list[client] for client in clients_this_round])
    global_auxiliary = None
    
    if fed_args.fed_alg == 'fednormbound': 
        for client in clients_this_round:
            local_update = local_dict_list[client]
            flattern = [local_update[key].view(-1) for key in global_dict.keys()]
            flat_update = torch.cat(flattern)

            update_norm = torch.norm(flat_update, p=2)
            
            norm_bound = fed_args.fed_normbounding
            sc_factor = norm_bound / update_norm if update_norm > norm_bound else 1.0

            for key in global_dict.keys():
                local_dict_list[client][key] *= sc_factor
        for key in global_dict.keys():
            global_dict[key] = sum([local_dict_list[client][key]* sample_num_list[client] / sample_this_round for client in clients_this_round])    
    else:  
        for key in global_dict.keys():
            global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])

            global_dict[key] = 0
            for client in clients_this_round:
                weight = sample_num_list[client] /sample_this_round
                global_dict[key] += local_dict_list[client][key]* weight

    return global_dict, global_auxiliary

def global_aggregate_encrypted(fed_args, global_dict, sample_num_list, clients_this_round, round_idx,global_encrypted_list, encrypted_dict_list, cc,keys, total_params, proxy_dict=None, opt_proxy_dict=None, auxiliary_info=None):
    sample_this_round = sum([sample_num_list[client] for client in clients_this_round])
    global_auxiliary = None

    if fed_args.fed_alg == 'fednormbound':
        normbound_plaintext = cc.MakeCKKSPackedPlaintext([1/fed_args.fed_normbounding]*(cc.GetRingDimension()//2))
        normbound_ciphertext = cc.Encrypt(keys, normbound_plaintext) 
        _, _, l2_norm = auxiliary_info

        for client in clients_this_round:
            ct = cc.EvalSub(normbound_ciphertext,l2_norm[client])
            polyF = computeF(1)
            polyG = [0, 2126 / pow(2.0, 10), 0,  -1359 / pow(2.0, 10)]

            d1 = 3.0
            d3 = -2.0

            ng = 3
            nf = 2

            for i in range(ng):
                ct = evaluate_comp(cc,polyG, ct)
            
            for i in range(nf):
                ct = evaluate_comp(cc,polyF, ct)

            relu_cts = trimmed_linear(ct,cc)
            scale_ct = cc.EvalMult(l2_norm[client],fed_args.fed_normbounding)
            cc.RelinearizeInPlace(scale_ct)
            cc.RescaleInPlace(scale_ct)
         
            scale_ct = cc.EvalAdd(scale_ct,-1)
            scale_mult = cc.EvalMult(scale_ct,relu_cts)
            cc.RelinearizeInPlace(scale_mult)
            cc.RescaleInPlace(scale_mult)
            scale_final = cc.EvalAdd(scale_mult,1)

            for cipher_index in range(len(encrypted_dict_list[clients_this_round[0]])):
                produce_gradient = cc.EvalMult(scale_final,encrypted_dict_list[client][cipher_index])
                cc.RelinearizeInPlace(produce_gradient)
                cc.RescaleInPlace(produce_gradient)
                encrypted_dict_list[client][cipher_index] = produce_gradient

        for cipher_index in range(len(encrypted_dict_list[clients_this_round[0]])):
            for client in clients_this_round:
                weight = sample_num_list[client] / sample_this_round
                weigted_ciphertext = cc.EvalMult(encrypted_dict_list[client][cipher_index],weight)

    else:
        for cipher_index in range(len(encrypted_dict_list[clients_this_round[0]])):
            for client in clients_this_round:
                weight = sample_num_list[client] / sample_this_round
                weigted_ciphertext = cc.EvalMult(encrypted_dict_list[client][cipher_index],weight)
                global_encrypted_list[cipher_index] = cc.EvalAdd(global_encrypted_list[cipher_index],weigted_ciphertext)

  
    return global_encrypted_list, global_auxiliary

def global_aggregate_encrypted_pairing(fed_args, global_dict, sample_num_list, clients_this_round, round_idx,global_encrypted_list, encrypted_dict_list, cc,keys, total_params, proxy_dict=None, opt_proxy_dict=None, auxiliary_info=None):
    sample_this_round = sum([sample_num_list[client] for client in clients_this_round])
    global_auxiliary = None

    if fed_args.fed_alg == 'fednormbound':
        normbound_plaintext = cc.MakeCKKSPackedPlaintext([1/fed_args.fed_normbounding]*(cc.GetRingDimension()//2))
        normbound_ciphertext = cc.Encrypt(keys, normbound_plaintext) 
        _, _, l2_norm = auxiliary_info

        for client in clients_this_round:
            ct = cc.EvalSub(normbound_ciphertext,l2_norm[client])
            polyF = computeF(1)
            polyG = [0, 2126 / pow(2.0, 10), 0,  -1359 / pow(2.0, 10)]
            d1 = 3.0
            d3 = -2.0
            ng = 3
            nf = 2

            for i in range(ng):
                ct = evaluate_comp(cc,polyG, ct)

            for i in range(nf):
                ct = evaluate_comp(cc,polyF, ct)

            relu_cts = trimmed_linear(ct,cc)
            scale_ct = cc.EvalMult(l2_norm[client],fed_args.fed_normbounding)
            cc.RelinearizeInPlace(scale_ct)
            cc.RescaleInPlace(scale_ct)

            scale_ct = cc.EvalAdd(scale_ct,-1)
            scale_mult = cc.EvalMult(scale_ct,relu_cts)
            cc.RelinearizeInPlace(scale_mult)
            cc.RescaleInPlace(scale_mult)
            scale_final = cc.EvalAdd(scale_mult,1)
            scale_final = cc.EvalMult(scale_final,10)

            for cipher_index in range(len(encrypted_dict_list[clients_this_round[0]])):
                produce_gradient = cc.EvalMult(scale_final,encrypted_dict_list[client][cipher_index])
                encrypted_dict_list[client][cipher_index] = produce_gradient

        for cipher_index in range(len(encrypted_dict_list[clients_this_round[0]])):
            for client in clients_this_round:
                weight_pair = round(sample_num_list[client] / sample_this_round * 10)
                weigted_ciphertext = cc.EvalMult(encrypted_dict_list[client][cipher_index],weight_pair)
                global_encrypted_list[cipher_index] = cc.EvalAdd(global_encrypted_list[cipher_index],weigted_ciphertext) 
    else:
        for cipher_index in range(len(encrypted_dict_list[clients_this_round[0]])):
            for client in clients_this_round:
                weight_pair = round(sample_num_list[client] / sample_this_round * 10)
                weigted_ciphertext = cc.EvalMult(encrypted_dict_list[client][cipher_index],weight_pair)
                global_encrypted_list[cipher_index] = cc.EvalAdd(global_encrypted_list[cipher_index],weigted_ciphertext)

    return global_encrypted_list, global_auxiliary



def exchange_indices(clients_this_round, sparseK_index, encrypted_index_list, global_index_list, cc):
    num_slots = len(sparseK_index[clients_this_round[0]])
    final_index = [0] * num_slots  
    for cipher_index in range(len(encrypted_index_list[clients_this_round[0]])):
        for client in clients_this_round:
            global_index_list[cipher_index] = cc.EvalAdd(global_index_list[cipher_index],encrypted_index_list[client][cipher_index])  

    return global_index_list