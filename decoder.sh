#! /bin/bash

model='treelstm_rel'
flag='treelstm_rel92.19'
data='subj'
model_path='save_models/2018-12-08-23-40-51-treelstm-model_epoch_4.pt'
save_dir='analyze/'

python3.5 decoder.py -seed 23 -gpu_device 0 -sort \
                              -dir ${data}_processed_data \
                              -model $model -model_path $model_path \
                              -type test -save_path ${save_dir}${model}.${data}.${flag}.wrong.txt