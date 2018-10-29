#! /bin/bash

python preprocess.py -raw_train_path ./data/stsa.fine.train -raw_dev_path ./data/stsa.fine.dev -raw_test_path ./data/stsa.fine.test \
                     -freq_vocab 0 -vcb_size 30000 -save_dir processed_data \
                     -shuffle false -seed 23 -gpu_seed 23 -use_cuda true