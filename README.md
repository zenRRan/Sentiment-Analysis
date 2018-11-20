## Introducion
- A classification of Sentiment Analysis which is implemented by pytorch.
* There are many data in [data](https://github.com/zenRRan/Sentiment-Analysis/tree/master/data), *.txt of that are came from [sent-conv-torch](https://github.com/harvardnlp/sent-conv-torch), *.conll.out of that are generated from our lab's parser.
    * CR
    * MPQA
    * TREC
    * MR
    * SUBJ
- My processed data by preprocessed.sh will be saved in [processed_data](https://github.com/zenRRan/Sentiment-Analysis/tree/master/processed_data).
* Trere are many models in [models](https://github.com/zenRRan/Sentiment-Analysis/tree/master/models).
    * Pooling
    * CNN
    * Multi_Channel_CNN
    * Multi_Layer_CNN
    * CharCNN
    * GRU
    * LSTM
    * LSTM_CNN
    * TreeLSTM
    * CNN_TreeLSTM
    * LSTM_TreeLSTM
    * Transformer(TODO)
- My log will be saved in [log](https://github.com/zenRRan/Sentiment-Analysis/tree/master/log).
* Trere are many scripts in [utils](https://github.com/zenRRan/Sentiment-Analysis/tree/master/utils)
    * Alphabet.py which is used to build dictionary.
    * Common.py which is saved unk-key and pad-key.
    * Embedding.py which is used to load pre_train embedding by Yang Song.
    * Evaluate.py which is used to calculate the F1.
    * Feature.py which is implemented a sentence's features, including word, word_id, label, root and so on.
    * build-batch.py which is used to build the data's mini batch.
    * log.py which is used to save the log.
    * opts.py
    *  which is implemented the argparses.
    * trainer.py which is used to train the data.
    * tree.py which is implemented the tree's methods.

## Requirement
        python : 3.5+
        pytorch : 0.4.0
        cuda : 8.0 (support GPU, you can choose)

## Usage
- first step

        sh preprocess.sh
- second step

       sh run.sh

## Result


| Data/Model        | TREC  | SUBJ  | MR    | CR    | MPQA  |
| ------            | ------ | ------ | ------ | ------ | ------ |
| Pooling           | 76.12 | 89.58 | 74.51 | 80 | 86.43 |
| CNN               | 90.4  | 91.98 | 77.73 | 84.38 | 88.96 |
| Char_CNN          | 91.52 | 93.33 | 78.91 | 84.38 | 86.33 |
| Multi_Channel_CNN | 89.06 | 94.06 | 79.59 | 83.12 | 88.48 |
| Multi_Layer_CNN   | 91.74 | 93.65 | 79    | 84.06 | 89.36 |
| LSTM              | 89.73 | 92.5  | 80.57 | 83.75 | 89.65 |
| LSTM_CNN          | 92.63 | 92.19 | 81.05 | 83.12 | 89.16 |
| GRU               | 89.06 | 92.6  | 79.1  | 83.44 | 89.75 |
| TreeLSTM          | 89.18 | 91.09 | 78.54 | 81.4  | 89.28 |
| CNN_TreeLSTM      | - | - | 79.11 | - | - |

## Question
If you have any questions about this code, please report to me immediately by `zenrran@qq.com`!