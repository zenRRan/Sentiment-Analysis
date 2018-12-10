#! /bin/bash

model='treelstm_rel'
flag='treelstm_rel92.19'
model_path='save_models/2018-12-09-10-18-14-treelstm_rel-model_epoch_5.pt'
save_dir='analyze/'

nohup python decoder.py -seed 23 -gpu_device 0 \
                              -dir processed_data \
                              -model $model -model_path $model_path \
                              -type test -save_path ${save_dir}${model}.${flag}.wrong.txt > \
                              $model.$model_path.log 2>&1 &