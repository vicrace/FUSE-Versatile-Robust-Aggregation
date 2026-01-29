max_steps=10
num_rounds=200
batch_size=16
gradient_accumulation_steps=1
seq_length=512
num_clients=10 
sample_clients=3
lora_r=32
lora_alpha=64  
lr=5e-5

dataset_name="sytelus/openwebtext" #lsb/openwebtext-all-minilm-l6-v2-embedding , sytelus/openwebtext
dataset_sample=40000 
model_name_or_path="openai-community/gpt2" #meta-llama/Llama-2-7b-hf"
output_dir=./output

gpu=2
fed_alg="fednormbound" #fedavg, fedprox, fednormbound

CUDA_VISIBLE_DEVICES=$gpu python plaintext_FL.py \
 --learning_rate $lr \
 --model_name_or_path $model_name_or_path \
 --dataset_name $dataset_name \
 --dataset_sample $dataset_sample \
 --fed_alg $fed_alg \
 --num_clients $num_clients \
 --sample_clients $sample_clients \
 --max_steps $max_steps \
 --num_rounds $num_rounds \
 --batch_size $batch_size \
 --gradient_accumulation_steps $gradient_accumulation_steps \
 --seq_length $seq_length \
 --peft_lora_r $lora_r \
 --peft_lora_alpha $lora_alpha \
 --use_peft \
 --output_dir $output_dir \
 --template "lambada" \
 --use_auth_token=True \
 --split_strategy "non-iid"