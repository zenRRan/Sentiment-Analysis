## Introduction
- A classification of Sentiment Analysis which is implemented by pytorch.
* There are many data in [data](https://github.com/zenRRan/Sentiment-Analysis/tree/master/data), *.txt of that are came from [sent-conv-torch](https://github.com/harvardnlp/sent-conv-torch), *.conll.out of that are generated from our lab's parser.
    * TREC
    * SUBJ
    * MR
    * CR
    * MPQA

- My processed data by preprocessed.sh will be saved in [processed_data](https://github.com/zenRRan/Sentiment-Analysis/tree/master/processed_data).
* There are many models in [models](https://github.com/zenRRan/Sentiment-Analysis/tree/master/models).
    * Pooling
    * CNN
    * Multi_Channel_CNN
    * Multi_Layer_CNN
    * CharCNN
    * GRU
    * LSTM
    * LSTM_CNN
    * TreeLSTM
    * biTreeLSTM
    * TreeLSTM_rel
    * biTreeLSTM_rel
    * CNN_TreeLSTM(ready to refresh)
    * LSTM_TreeLSTM(ready to refresh)
    * Transformer(TODO)
- My log will be saved in [log](https://github.com/zenRRan/Sentiment-Analysis/tree/master/log).
* There are many scripts in [utils](https://github.com/zenRRan/Sentiment-Analysis/tree/master/utils).
    * Alphabet.py which is used to build dictionary.
    * Common.py which is saved unk-key and pad-key.
    * Embedding.py which is used to load pre_train embedding by Yang Song.
    * Evaluate.py which is used to calculate the F1.
    * Feature.py which is implemented a sentence's features, including word, word_id, label, root and so on.
    * build-batch.py which is used to build the data's mini batch.
    * log.py which is used to save the log.
    * opts.py which is implemented the argparses.
    * trainer.py which is used to train the data.
    * tree.py which is implemented the tree's methods.

## Requirement
        python : 3.5+
        pytorch : 0.4.1
        cuda : 8.0 (support GPU, you can choose)

## Usage
- first step

        sh preprocess.sh
- second step

        sh run.sh
- third step (also called decoder step which will output a file whose predictions were wrong. If necessary)

        sh decoder.sh

## Result

| Data/Model(acc)   | TREC  | SUBJ  | MR    | CR    | MPQA  |
| ------            | ----- | ----- | ----- | ----- | ----- |
| Pooling           | 76.12 | 92.10 | 75.92 | 79.03 | 85.97 |
| CNN               | 91.40 | 93.20 | 77.05 | 83.60 | 88.34 |
| Char_CNN          | 92.20 | 93.30 | 78.66 | 83.60 | 88.25 |
| Multi_Channel_CNN | 89.20 | 93.40 | 78.56 | 81.45 | 88.06 |
| Multi_Layer_CNN   | 91.00 | 93.70 | 78.28 | 83.06 | 88.44 |
| LSTM              | 89.20 | 92.50 | 78.94 | 81.99 | 89.57 |
| LSTM_CNN          | 90.08 | 93.40 | 79.51 | 82.80 | 88.82 |
| GRU               | 89.40 | 92.80 | 78.28 | 82.26 | 89.48 |
| TreeLSTM          | 89.60 | 92.60 | 79.79 | 84.41 | 88.91 |
| biTreeLSTM        | 90.40 | 92.70 | 79.13 | 83.87 | 88.91 |
| TreeLSTM_rel      | 91.29 | 92.20 | 80.36 | 82.26 | 89.06 |
| biTreeLSTM_rel    | 91.20 | 92.80 | 80.26 | 83.60 | 89.10 |
| CNN_TreeLSTM      | - | - | - | - | - |
| LSTM_TreeLSTM     | - | - | - | - | - |

### In addition:

#### Data
 - TREC: `TREC question dataset` - task involves classifying a question into 6 question types (whether the question is about abbreviation, entity, description, human, location and numeric value) (Li and Roth, 2002).
 - SUBJ: `Subjectivity dataset` where the task is to classify a sentence as being subjective or objective (Pang and Lee, 2004).
 - MR: `Movie reviews` with one sentence per review. Classification involves detecting positive/negative reviews (Pang and Lee, 2005).
 - CR: `Customer reviews` of various products (cameras, MP3s etc.). Task is to predict positive/negative reviews (Hu and Liu, 2004).
 - MPQA: `The MPQA Opinion Corpus` contains news articles from a wide variety of news sources manually annotated for opinions and other private states (i.e., beliefs, emotions, sentiments, speculations, etc.).

#### Emphasize
 - `pre_trained_embed` which is using `glove.6B.100d.txt`.
 - `TreeLSTM` which is using `ChildSum` method.

#### Future Work
 Other methods about `TreeLSTM` will be updated in the near future.

## Question
Glad to receive your report by `zenrran@qq.com`, If you have any questions about this code !