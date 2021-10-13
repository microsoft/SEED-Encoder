More details can be found in https://github.com/playing-code/fairseq .

Data processing: you need to check the file path and name

```
python data_process.py
```


## Training command: change data dir and files to yours

## For training, to use different part for candidate news like using new category and abstract, need to set "--field cat_abs".

```
python train_plain_bert_dot4_fp16.py --data_dir $DATA_DIR --save_dir $SAVE_DIR \
--data_file train_ms_roberta_cat_abs_shuffle_fast.txt --feature_file news_token_features_roberta_abs_cat_fast.txt \
--size 8 --batch_size 1024 --log_file log_dot4 --test_data_file dev_fast \
--test_feature_file news_token_features_roberta_abs_cat_dev_fast.txt --field cat_abs --gpu_size 24 --model_file $Load_file --eval_step $7
```

## An example for test


```
python test_plain_bert_dot4.py --data_dir $DATA_DIR --save_dir $SAVE_DIR --data_file dev_fast20.txt --feature_file news_token_features_roberta_abs_cat_fast.txt --cudaid 0 --model_file $Load_file --log_file dev_res_${prefix}_20.txt --can_length 20 --gpu_size 60 --field cat_abs \
--model_version dot4 &

python test_plain_bert_dot4.py --data_dir $DATA_DIR --save_dir $SAVE_DIR --data_file dev_fast40.txt --feature_file news_token_fnews_token_features_roberta_abs_cat_fasteatures_roberta_abstract_dev_fast.txt --cudaid 1 --model_file $Load_file --log_file dev_res_${prefix}_40.txt --can_length 40 --gpu_size 30 --field cat_abs \
--model_version dot4 &

python test_plain_bert_dot4.py --data_dir $DATA_DIR --save_dir $SAVE_DIR --data_file dev_fast60.txt --feature_file news_token_features_roberta_abs_cat_fast.txt --cudaid 2 --model_file $Load_file --log_file dev_res_${prefix}_60.txt --can_length 60 --gpu_size 20 --field cat_abs \
--model_version dot4 &

python test_plain_bert_dot4.py --data_dir $DATA_DIR --save_dir $SAVE_DIR --data_file dev_fast90.txt --feature_file news_token_features_roberta_abs_cat_fast.txt --cudaid 3 --model_file $Load_file --log_file dev_res_${prefix}_90.txt --can_length 90 --gpu_size 15 --field cat_abs \
--model_version dot4 &
```