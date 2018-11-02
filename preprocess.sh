#! /bin/bash

python preprocess.py -raw_train_path ./data/MR/mr.train.txt \
                     -raw_dev_path ./data/MR/mr.dev.txt \
                     -raw_test_path ./data/MR/mr.test.txt \
                     -freq_vocab 0 -vcb_size 30000 -save_dir processed_data \
                     -shuffle false -seed 23 -gpu_seed 23 -use_cuda true