#!/usr/bin/env bash
export DATA_PATH=../data

CUDA_VISIBLE_DEVICES=0 python test.py --run_mode test \
--local_rank -1 \
--model_class bert2span \
--pretrain_model_type roberta-base \
--dataset_class kp20k \
--per_gpu_test_batch_size 64 \
--preprocess_folder $DATA_PATH/prepro_dataset \
--pretrain_model_path $DATA_PATH/pretrain_model \
--cached_features_dir $DATA_PATH/cached_features \
--eval_checkpoint /home/banifeng/workspace/code_repo/BERT-KPE/results/train_bert2span_kp20k_roberta_08.08_13.57/checkpoints/bert2span.kp20k.roberta.epoch_5.checkpoint