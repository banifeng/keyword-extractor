#!/usr/bin/env bash
export MAX_LENGTH=510
export BERT_MODEL=/home/whou/workspace/bert-base-uncased

#cat train.txt dev.txt test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > labels.txt
export OUTPUT_DIR=ner-model
export BATCH_SIZE=32
export NUM_EPOCHS=5
export SAVE_STEPS=750
export SEED=1

python3 bert_extractor.py --data_dir ./semeval_data/ \
--model_type bert \
--labels ./semeval_data/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
#--no_cuda
#--do_predict
