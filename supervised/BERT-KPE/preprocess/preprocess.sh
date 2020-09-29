export DATA_PATH=../data

# preprocess openkp or kp20k
python preprocess.py --dataset_class kp20k \
--source_dataset_dir $DATA_PATH/dataset \
--output_path $DATA_PATH/prepro_dataset