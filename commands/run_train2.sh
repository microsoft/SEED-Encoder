#!/bin/bash
#
# This script is for training with updated ann driver
#
# The design for this ann driver is to have 2 separate processes for training: one for passage/query 
# inference using trained checkpoint to generate ann data and calcuate ndcg, another for training the model 
# using the ann data generated. Data between processes is shared on common directory, model_dir for checkpoints
# and model_ann_data_dir for ann data.
#
# This script initialize the training and start the model training process
# It first preprocess the msmarco data into indexable cache, then generate a single initial ann data
# version to train on, after which it start training on the generated ann data, continously looking for
# newest ann data generated in model_ann_data_dir
#
# To start training, you'll need to run this script first
# after intial ann data is created (you can tell by either finding "successfully created 
# initial ann training data" in console output or if you start seeing new model on tensorboard),
# start run_ann_data_gen.sh in another dlts job (or same dlts job using split GPU)
#
# Note if preprocess directory or ann data directory already exist, those steps will be skipped
# and training will start immediately

# Passage ANCE(FirstP) 
gpu_no=4
seq_length=512
model_type=rdot_nll_fairseq_fast
tokenizer_type="roberta-base-fast-doceval_dev"
# tokenizer_type="roberta-base-fast-docdev2"
base_data_dir="../../data/raw_data/"
preprocessed_data_dir="${base_data_dir}ann_data_${tokenizer_type}_${seq_length}temp/"
job_name="OSPass512"
pretrained_checkpoint_dir="warmup or trained checkpoint path"
data_type=0
warmup_steps=5000
per_gpu_train_batch_size=8
gradient_accumulation_steps=2
learning_rate=1e-6

# Document ANCE(FirstP) 
# gpu_no=2
# seq_length=512
# tokenizer_type="roberta-base-fast-doc"
# model_type=rdot_nll_fairseq_fast
# base_data_dir="../../data/raw_data/"
# preprocessed_data_dir="${base_data_dir}ann_data_${tokenizer_type}_${seq_length}/"
# job_name="exp_12_02_04"
# # pretrained_checkpoint_dir='../../data/model_temp/model.pt'
# pretrained_checkpoint_dir="${base_data_dir}exp_test/checkpoint-20/model.pt"
# data_type=0
# warmup_steps=3000
# per_gpu_train_batch_size=4
# gradient_accumulation_steps=2
# learning_rate=5e-6

# # Document ANCE(MaxP) 
# gpu_no=1
# seq_length=2048
# tokenizer_type="roberta-base-fast"
# model_type=rdot_nll_multi_chunk_fairseq_fast
# base_data_dir="../../data/raw_data/"
# preprocessed_data_dir="${base_data_dir}ann_data_${tokenizer_type}_${seq_length}/"
# job_name="OSDoc2048"
# pretrained_checkpoint_dir=../../data/model_temp/roberta_dr_140.pt
# # pretrained_checkpoint_dir=../../data/model_temp/model.pt
# data_type=0
# warmup_steps=500
# per_gpu_train_batch_size=2
# gradient_accumulation_steps=8
# learning_rate=1e-5

# ##################################### Data Preprocessing ################################
# blob_model_dir="${base_data_dir}${job_name}/"
# blob_model_ann_data_dir="${blob_model_dir}ann_data/"

# model_dir="${base_data_dir}exp_test/"
# model_ann_data_dir="${model_dir}ann_data/"

preprocess_cmd="\
python ../data/msmarco_data.py --data_dir $base_data_dir --out_data_dir $preprocessed_data_dir --train_model_type $model_type  --max_seq_length $seq_length --data_type $data_type --bpe_vocab_file ../../data/bert-16g-0930/vocab.txt\
"

echo $preprocess_cmd
eval $preprocess_cmd

# if [[ $? = 0 ]]; then
#     echo "successfully created preprocessed data"
# else
# 	echo "preprocessing failed"
#     echo "failure: $?"
#     exit 1
# fi

##################################### Inital ANN Data generation ################################
# initial_data_gen_cmd="\
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=$gpu_no ../drivers/run_ann_data_gen.py --training_dir $model_dir \
# --init_model_dir $pretrained_checkpoint_dir --train_model_type $model_type --output_dir $model_ann_data_dir \
# --cache_dir "${model_ann_data_dir}cache/" --data_dir $preprocessed_data_dir --max_seq_length $seq_length \
# --per_gpu_eval_batch_size 16 --topk_training 200 --negative_sample 20 --bpe_vocab_file ../../data/bert-16g-0930/vocab.txt &
# "

# echo $initial_data_gen_cmd
# eval $initial_data_gen_cmd 

# if [[ $? = 0 ]]; then
#     echo "successfully created initial ann training data"
# else
# 	echo "initial data generation failed"
#     echo "failure: $?"
#     exit 1
# fi

# ############################################# Training ########################################
# train_cmd="\
# CUDA_VISIBLE_DEVICES=2,3  python -m torch.distributed.launch --nproc_per_node=$gpu_no --master_addr 127.0.0.2 --master_port 29501 ../drivers/run_ann.py --train_model_type $model_type \
# --model_name_or_path $pretrained_checkpoint_dir --task_name MSMarco --triplet --data_dir $preprocessed_data_dir \
# --ann_dir $model_ann_data_dir --max_seq_length $seq_length --per_gpu_train_batch_size=$per_gpu_train_batch_size \
# --gradient_accumulation_steps $gradient_accumulation_steps --learning_rate $learning_rate --output_dir $model_dir \
# --warmup_steps $warmup_steps --logging_steps 1 --save_steps 10 --optimizer lamb --single_warmup --bpe_vocab_file ../../data/bert-16g-0930/vocab.txt \
# --blob_ann_dir $blob_model_ann_data_dir --blob_output_dir $blob_model_dir &" 

# echo $train_cmd
# eval $train_cmd


# wait


# gpu_no=4
# seq_length=512
# model_type=rdot_nll_fairseq_fast
# tokenizer_type="roberta-base-fast-doc"
# base_data_dir="../../data/raw_data/"
# preprocessed_data_dir="${base_data_dir}ann_data_${tokenizer_type}_${seq_length}/"



# pretrained_checkpoint_dir="../../data/model_temp/roberta_decode_document.pt"
# data_type=0
# # warmup_steps=5000
# # per_gpu_train_batch_size=8
# # gradient_accumulation_steps=2
# # learning_rate=1e-6



# model_dir="${base_data_dir}test_roberta_decode_doc/"
# model_ann_data_dir="${model_dir}ann_data/"

# initial_data_gen_cmd="\
# python -m torch.distributed.launch --nproc_per_node=$gpu_no ../drivers/run_ann_data_gen.py --training_dir $model_dir \
# --init_model_dir $pretrained_checkpoint_dir --train_model_type $model_type --output_dir $model_ann_data_dir \
# --cache_dir "${model_ann_data_dir}cache/" --data_dir $preprocessed_data_dir --max_seq_length $seq_length \
# --per_gpu_eval_batch_size 256 --topk_training 200 --negative_sample 20 --end_output_num 0 --inference --bpe_vocab_file ../../data/bert-16g-0930/vocab.txt \
# "

# echo $initial_data_gen_cmd
# eval $initial_data_gen_cmd