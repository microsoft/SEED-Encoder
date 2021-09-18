This repository provides the fine-tuning stage on Marco ranking task for [SEED-Encoder](https://arxiv.org/abs/2102.09206) and is based on ANCE (https://github.com/microsoft/ANCE).

# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.4.0
* Python version >= 3.6
* Please install Apex with CUDA and C++ extensions ([apex github](https://github.com/NVIDIA/apex)).


# Fine-tuning for SEED-Encoder
* We follow the ranking experiments in ANCE ([Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval](https://arxiv.org/pdf/2007.00808.pdf) ) as our downstream tasks.

## Requirements

To install requirements, run the following commands:

```setup
git clone https://github.com/microsoft/SEED-Encoder
cd SEED-Encoder
python setup.py install
```

## Data Download
To download all the needed data, run:
```
bash commands/data_download.sh 
```

## Our Checkpoints
[Pretrained SEED-Encoder with 3-layer decoder, attention span = 2 ](https://fastbertjp.blob.core.windows.net/release-model/SEED-Encoder-3-decoder-2-attn.pt)

[Pretrained SEED-Encoder with 1-layer decoder, attention span = 8 ](https://fastbertjp.blob.core.windows.net/release-model/SEED-Encoder-1-decoder-8-attn.pt)

[SEED-Encoder warmup checkpoint](https://fastbertjp.blob.core.windows.net/release-model/SEED-Encoder-warmup-90000.pt)

[ANCE finetuned SEED-Encoder checkpoint on passage ranking task](https://fastbertjp.blob.core.windows.net/release-model/SEED-Encoder-pass-440000.pt)

[ANCE finetuned SEED-Encoder checkpoint on document ranking task](https://fastbertjp.blob.core.windows.net/release-model/SEED-Encoder-doc-800000.pt)

[bpe file used in our tokenizer](https://fastbertjp.blob.core.windows.net/release-model/vocab.txt)

[DPR finetuned SEED-Encoder checkpoint on NQ task](https://fastbertjp.blob.core.windows.net/release-model/dpr_biencoder.37)

[ANCE finetuned SEED-Encoder checkpoint on NQ task](https://fastbertjp.blob.core.windows.net/release-model/ance-nq-checkpoint)




## Data Preprocessing
The command to preprocess passage and document data is listed below:

```
python data/msmarco_data.py \
--data_dir $raw_data_dir \
--out_data_dir $preprocessed_data_dir \ 
--train_model_type {use rdot_nll_fairseq_fast for SEED-Encoder ANCE FirstP} \ 
--max_seq_length {use 512 for ANCE FirstP, 2048 for ANCE MaxP} \ 
--data_type {use 1 for passage, 0 for document}
--bpe_vocab_file $bpe_vocab_file
```

The data preprocessing command is included as the first step in the training command file commands/run_train.sh

## Warmup for Training
        model_file=SEED-Encoder-3-decoder-2-attn.pt
        vocab=vocab.txt

        python3 -m torch.distributed.launch --nproc_per_node=8 ../drivers/run_warmup.py \
        --train_model_type rdot_nll_fairseq_fast --model_name_or_path $LOAD_DIR --model_file $model_file --task_name MSMarco --do_train \
        --evaluate_during_training --data_dir $DATA_DIR \
        --max_seq_length 128 --per_gpu_eval_batch_size=256  --per_gpu_train_batch_size=32 --learning_rate 2e-4 --logging_steps 100 --num_train_epochs 2.0 \
        --output_dir $SAVE_DIR --warmup_steps 1000 --overwrite_output_dir --save_steps 10000 --gradient_accumulation_steps 1 --expected_train_size 35000000 \
        --logging_steps_per_eval 100 --fp16 --optimizer lamb --log_dir $SAVE_DIR/log --bpe_vocab_file $vocab

## ANCE Training (passage, you may first use the second command to generate the initial data)

        gpu_no=4
        seq_length=512
        tokenizer_type="roberta-base-fast"
        model_type=rdot_nll_fairseq_fast
        base_data_dir={}
        preprocessed_data_dir="${base_data_dir}ann_data_${tokenizer_type}_${seq_length}/"
        job_name=$exp_name
        pretrained_checkpoint_dir=SEED-Encoder-warmup-90000.pt
        data_type=1
        warmup_steps=5000
        per_gpu_train_batch_size=16
        gradient_accumulation_steps=1
        learning_rate=1e-6
        vocab=vocab.txt

        blob_model_dir="${base_data_dir}${job_name}/"
        blob_model_ann_data_dir="${blob_model_dir}ann_data/"

        model_dir="./${job_name}/"
        model_ann_data_dir="${model_dir}ann_data/"

        
        CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=$gpu_no ../drivers/run_ann_data_gen.py --training_dir $model_dir \
        --init_model_dir $pretrained_checkpoint_dir --train_model_type $model_type --output_dir $model_ann_data_dir \
        --cache_dir {} --data_dir $preprocessed_data_dir --max_seq_length $seq_length \
        --per_gpu_eval_batch_size 64 --topk_training 200 --negative_sample 20 --bpe_vocab_file $vocab




        CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$gpu_no --master_addr 127.0.0.2 --master_port 35000 ../drivers/run_ann.py --train_model_type $model_type \
        --model_name_or_path $pretrained_checkpoint_dir --task_name MSMarco --triplet --data_dir $preprocessed_data_dir \
        --ann_dir $model_ann_data_dir --max_seq_length $seq_length --per_gpu_train_batch_size=$per_gpu_train_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps --learning_rate $learning_rate --output_dir $model_dir \
        --warmup_steps $warmup_steps --logging_steps 100 --save_steps 10000 --optimizer lamb --single_warmup --bpe_vocab_file $vocab \
        --blob_ann_dir $blob_model_ann_data_dir --blob_output_dir $blob_model_dir
       
## ANCE Training (document)

        gpu_no=4
        seq_length=512
        tokenizer_type="roberta-base-fast-docdev2"
        model_type=rdot_nll_fairseq_fast
        base_data_dir={}
        preprocessed_data_dir="${base_data_dir}ann_data_${tokenizer_type}_${seq_length}/"
        job_name=$exp_name
        pretrained_checkpoint_dir=SEED-Encoder-warmup-90000.pt
        data_type=0
        warmup_steps=3000
        per_gpu_train_batch_size=4
        gradient_accumulation_steps=4
        learning_rate=5e-6
        vocab=vocab.txt

        blob_model_dir="${base_data_dir}${job_name}/"
        blob_model_ann_data_dir="${blob_model_dir}ann_data/"

        model_dir="./${job_name}/"
        model_ann_data_dir="${model_dir}ann_data/"

        CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=$gpu_no ../drivers/run_ann_data_gen.py --training_dir $model_dir \
        --init_model_dir $pretrained_checkpoint_dir --train_model_type $model_type --output_dir $model_ann_data_dir \
        --cache_dir {} --data_dir $preprocessed_data_dir --max_seq_length $seq_length \
        --per_gpu_eval_batch_size 16 --topk_training 200 --negative_sample 20 --bpe_vocab_file $vocab



        CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$gpu_no --master_addr 127.0.0.2 --master_port 35000 ../drivers/run_ann.py --train_model_type $model_type \
        --model_name_or_path $pretrained_checkpoint_dir --task_name MSMarco --triplet --data_dir $preprocessed_data_dir \
        --ann_dir $model_ann_data_dir --max_seq_length $seq_length --per_gpu_train_batch_size=$per_gpu_train_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps --learning_rate $learning_rate --output_dir $model_dir \
        --warmup_steps $warmup_steps --logging_steps 100 --save_steps 10000 --optimizer lamb --single_warmup --bpe_vocab_file $vocab \
        --blob_ann_dir $blob_model_ann_data_dir --blob_output_dir $blob_model_dir --cache_dir {}

## To reproduce our results you can use our checkpoints to generate the embeddings and then evaluate the results:
        
        python -m torch.distributed.launch --nproc_per_node=$gpu_no ../drivers/run_ann_data_gen.py --training_dir $model_dir \
        --init_model_dir $pretrained_checkpoint_dir --train_model_type $model_type --output_dir $blob_model_ann_data_dir \
        --cache_dir {} --data_dir $preprocessed_data_dir --max_seq_length $seq_length \
        --per_gpu_eval_batch_size 64 --topk_training 200 --negative_sample 20 --end_output_num 0 --inference --bpe_vocab_file $vocab
        
        
        python ../evaluation/eval.py


## NQ scripts

    The running script is in commands/run_ann_data_gen_dpr.sh and commands/run_tran_dpr.sh



## Results of SEED-Encoder

|   MSMARCO Dev Passage Retrieval    | MRR@10  | Recall@1k |
|------------------------------|---------------|--------------------- |
| BM25 warmup checkpoint     |     0.329     |      0.953     |
| ANCE Passage  checkpoint  |     0.334   |   0.961       |

|   MSMARCO Document Retrieval    | MRR@10 (Dev)  |  MRR@10 (Eval) |
|---------------- | -------------- | -------------- |
|  ANCE Document (FirstP)  checkpoint   |     0.394       |    0.362     | 


| NQ Task      | Top-1  |  Top-5    | Top-20  |  Top-100 |  MRR@20    | P@20  |
|---------------- | -------------- | -------------- |-------------- | -------------- | -------------- |-------------- |
| DPR checkpoint    |     46.1       |        68.8     |    80.4     |   87.1      |   56.2    |    20.1   |
| ANCE NQ checkpoint    |   52.5        |       73.1      |      83.1   |   88.7   |       61.5   |    22.5



## Our huggingface Checkpoints
[Pretrained SEED-Encoder with 3-layer decoder, attention span = 2 ](https://fastbertjp.blob.core.windows.net/release-model/SEED-Encoder-3-decoder-layers.tar)

[Pretrained SEED-Encoder with 1-layer decoder, attention span = 8 ](https://fastbertjp.blob.core.windows.net/release-model/SEED-Encoder-1-decoder-layer.tar)

[SEED-Encoder warmup checkpoint](https://fastbertjp.blob.core.windows.net/release-model/SEED-Encoder-warmup-90000.tar)

[ANCE finetuned SEED-Encoder checkpoint on passage ranking task](https://fastbertjp.blob.core.windows.net/release-model/SEED-Encoder-pass-440000.tar)

[ANCE finetuned SEED-Encoder checkpoint on document ranking task](https://fastbertjp.blob.core.windows.net/release-model/SEED-Encoder-doc-800000.tar)



## Load the huggingface checkpoints and run


    DATA_DIR=../../data/raw_data
    SAVE_DIR=../../temp/
    LOAD_DIR=$your_dir/SEED-Encoder-warmup-90000/

    python3 -m torch.distributed.launch --nproc_per_node=8 ../drivers/run_warmup.py \
    --train_model_type seeddot_nll --model_name_or_path $LOAD_DIR --task_name MSMarco --do_train \
    --evaluate_during_training --data_dir $DATA_DIR \
    --max_seq_length 128 --per_gpu_eval_batch_size=512  --per_gpu_train_batch_size=2 --learning_rate 2e-4 --logging_steps 1 --num_train_epochs 2.0 \
    --output_dir $SAVE_DIR --warmup_steps 1000 --overwrite_output_dir --save_steps 1 --gradient_accumulation_steps 1 --expected_train_size 35000000 \
    --logging_steps_per_eval 1 --fp16 --optimizer lamb --log_dir $SAVE_DIR/log --do_lower_case --fp16
