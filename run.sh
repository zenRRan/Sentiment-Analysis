#! /bin/sh

model='treelstm_rel'
data='cr'
nohup python3.5 -u main.py -model $model -data_dir ${data}_processed_data \
       -train_batch_size 64 -train_batch_type same \
       -seed 23 -gpu_seed 23 -gpu_device 0 \
       -save_model_dir save_models \
       -embed_size 100 -char_embed_size 50  -embed_uniform_init 0 \
       -pre_embed_path /home/zrr/projects/Data/glove.6B/glove.6B.100d.txt \
       -kernel_num 100 -stride 1 \
       -hidden_size 100 -hidden_num 1 -dropout 0.3 -bidirectional \
       -optim adam -lr 0.001 -momentum 0.9 -weight_decay 1e-8 \
       -epoch 100 -print_every 10 -shuffer -sort \
       -log_dir log -log_fname ${model}.$data.log > ${model}.$data.log 2>&1 &

tail -f log/${model}.$data.log
