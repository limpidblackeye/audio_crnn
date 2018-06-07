#!/bin/bash 

TEST_WAV_DIR="../../data/audio_test"
TRAIN_WAV_DIR="../../data/audio_train"
# EVALUATION_WAV_DIR="/vol/vssp/datasets/audio/audioset/task4_dcase2017_audio/official_downloads/evaluation"

# ./features_logmel/train

# You can to modify to your own workspace. 
WORKSPACE=`pwd`

# Extract features
python data_prepare.py extract_features --wav_dir=$TEST_WAV_DIR --out_dir="./features/logmel/test" --recompute=True
python data_prepare.py extract_features --wav_dir=$TRAIN_WAV_DIR --out_dir="./features/logmel/train" --recompute=True
# python prepare_data.py extract_features --wav_dir=$EVALUATION_WAV_DIR --out_dir=$WORKSPACE"/features/logmel/evaluation" --recompute=True

# Pack features
python data_prepare.py pack_features --fe_dir=$WORKSPACE"/features/logmel/testing" --csv_path="meta_data/testing_set.csv" --out_path=$WORKSPACE"/packed_features/logmel/testing.h5"
python data_prepare.py pack_features --fe_dir=$WORKSPACE"/features/logmel/training" --csv_path="meta_data/training_set.csv" --out_path=$WORKSPACE"/packed_features/logmel/training.h5"
python data_prepare.py pack_features --fe_dir=$WORKSPACE"/features/logmel/evaluation" --csv_path="" --out_path=$WORKSPACE"/packed_features/logmel/evaluation.h5"

# Calculate scaler
python data_prepare.py calculate_scaler --hdf5_path=$WORKSPACE"/packed_features/logmel/training.h5" --out_path=$WORKSPACE"/scalers/logmel/training.scaler"

