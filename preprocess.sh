big_doc='CR'
file='cr'
python3.5 preprocess.py -raw_train_path ./data/$big_doc/$file.train.txt \
                        -raw_dev_path ./data/$big_doc/$file.dev.txt \
                        -raw_test_path ./data/$big_doc/$file.test.txt \
                        -train_conll_path ./data/$big_doc/$file.train.txt.conll.out \
                        -test_conll_path ./data/$big_doc/$file.test.txt.conll.out \
                        -dev_conll_path ./data/$big_doc/$file.dev.txt.conll.out \
                        -freq_vocab 0 -vcb_size 30000 -save_dir ${file}_processed_data \
                        -shuffle false -seed 23 -use_cuda true
