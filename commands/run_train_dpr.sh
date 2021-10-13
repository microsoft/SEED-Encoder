gpu_no=8

# model type
model_type="dpr_fast"
seq_length=256
triplet="--triplet --optimizer lamb" # set this to empty for non triplet model
out_data_dir="../../data/raw_data/QA_NQ_data/" # change this for different data_type

# hyper parameters
train_batch_size=32
gradient_accumulation_steps=1
learning_rate=1e-5
warmup_steps=1000

# input/output directories
preprocessed_data_dir="../../data/raw_data/QA_NQ_data/" 
base_data_dir="${out_data_dir}"
#job_name="exp_21_04_22_01"
job_name="exp_21_09_22_02"
#model_dir="./"
#model_ann_data_dir="${model_dir}ann_data/"

model_dir="${base_data_dir}${job_name}/"
model_ann_data_dir="${model_dir}ann_data/"
pretrained_checkpoint_dir="../../data/model_temp/dpr_biencoder.37"
#pretrained_checkpoint_dir="../../data/raw_data/QA_NQ_data/exp_21_09_22_01/checkpoint-175000"
#pretrained_checkpoint_dir="../../data/raw_data/QA_NQ_data/exp_21_09_22_01/checkpoint-270000"
#pretrained_checkpoint_dir="../../data/raw_data/QA_NQ_data/exp_21_09_22_02/checkpoint-40000"

#blob_output_dir='./'
#blob_ann_dir="${blob_output_dir}ann_data/"

train_cmd="\
python -m torch.distributed.launch --nproc_per_node=$gpu_no --master_addr 127.0.0.2 --master_port 35000 ../drivers/run_ann_dpr.py --model_type $model_type \
--model_name_or_path $pretrained_checkpoint_dir --task_name MSMarco $triplet --data_dir $preprocessed_data_dir \
--ann_dir $model_ann_data_dir --max_seq_length $seq_length --per_gpu_train_batch_size=$train_batch_size \
--gradient_accumulation_steps $gradient_accumulation_steps --learning_rate $learning_rate --output_dir $model_dir \
--warmup_steps $warmup_steps --logging_steps 100 --save_steps 5000 --log_dir $model_dir/log/ "



echo $train_cmd
eval $train_cmd

# echo "copy current script to model directory"
# sudo cp $0 $model_dir