#! /bin/sh

python -u main.py -model cnn -data_dir processed_data \
       -train_batch_size 64 -train_batch_type normal \
       -seed 23 -gpu_seed 23 -gpu_device 0 \
       -save_model_dir save_models \
       -embed_size 100 -char_embed_size 50  -embed_uniform_init 0 -embed_dropout 0 \
       -kernel_num 100 -stride 1 \
       -hidden_size 100 -hidden_num 1 -hidden_dropout 0 -bidirectional \
       -optim adam -lr 0.001 -momentum 0.9 -fc_dropout 0.1 -weight_decay 1e-8 \
       -epoch 100 -print_every 10 -shuffer -sort \
       -log_dir log -log_fname pooling.log