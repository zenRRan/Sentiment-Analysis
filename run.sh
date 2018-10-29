#! /bin/bash

python -u main.py -model pooling -data_dir processed_data \
       -train_batch_size 64 -train_batch_type normal \
       -seed 23 -use_cuda true -gpu_seed 23 -gpu_device 0 \
       -save_model_dir save_models \
       -embed_size 100 -char_embed_size 50 -pre_embed_path -embed_uniform_init 0 -embed_dropout 0 \
       -kernel_size [3,5,7] -kernel_num 100 -stride 1 \
       -hidden_size 100 -hidden_num 1 -hidden_dropout 0 -bidirectional true \
       -optim adam -lr 0.001 -momentum 0.9 -fc_dropout 0 -weight_decay 1e-8 \
       -epoch 100 -print_every 10 -shuffer -sort \
       -log_dir log -log_fname pooling.log