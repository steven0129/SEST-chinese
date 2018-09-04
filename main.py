#coding=utf8
import matplotlib
matplotlib.use('Agg')

import os
import torch
import fire
import ltp
import myDict
import csv
import codecs # To open the file in specific mode
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import Env
from RNN import BiLSTM

options = Env()

# Global Variable
DICT_DIR = 'dict'
DATA_DIR = 'data'

def dictGen(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(options, k_, v_)

    myDict.gen(DICT_DIR)

def skipgram(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(options, k_, v_)

    S = ltp.Sentence()
    W = ltp.Word(dictDir=DICT_DIR)

    with open(f'{DATA_DIR}/sky_dragon.txt') as f:
        contents = filter(lambda x: not x.startswith('['), f.readlines())
        contents = map(lambda x: x.rstrip('\n'), list(contents))
        content = ''.join(list(contents))

    print('斷句中...')
    sents = S.split(content)

    print('斷詞中...')
    wordLists = list(map(W.split, tqdm(sents)))

    with open('tmp.txt', 'w') as f:
        tmpWordLists = []
        STRING_LEN = 30
        
        for words in wordLists:
            tmpWords = words
            tmpWords.append('<end>')
            if len(words) <= STRING_LEN + 1:
                for i in range(STRING_LEN + 1 - len(tmpWords)):
                    tmpWords.append('<empty>')
            else:
                tmpWords = tmpWords[:STRING_LEN + 1]

            tmpWordLists.append(tmpWords)

        tmpSents = list(map(lambda x: ' '.join(x), tmpWordLists))
        tmpSents = list(map(lambda x: f'<start> {x}', tmpSents))
        tmpStr = '\n'.join(tmpSents)
        f.write(tmpStr)
        os.system(f'./fasttext skipgram -input tmp.txt -output {options.sgm_result}/skipgram -dim {options.word_dim} -epoch {options.sgm_epochs} -ws {options.sgm_ws} -lrUpdateRate {options.sgm_lr_update_rate} -thread {options.thread}')
        os.remove('./tmp.txt')

def SEST(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(options, k_, v_)

    S = ltp.Sentence()
    W = ltp.Word(dictDir=DICT_DIR)

    with open(f'{DATA_DIR}/sky_dragon.txt') as f:
        contents = filter(lambda x: not x.startswith('['), f.readlines())
        contents = map(lambda x: x.rstrip('\n'), list(contents))
        content = ''.join(list(contents))

    print('斷句中...')
    sents = S.split(content)

    print('斷詞中...')
    wordLists = list(map(W.split, tqdm(sents)))

    print('依存關係分析中...')
    dependencies = list(map(S.parse, tqdm(wordLists)))

    with open('sentvec/relations.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for idx, dep in enumerate(dependencies):
            nums = []
            relations = []

            for index, (head, relation) in enumerate(dep):
                nums.append(f'{head}->{index + 1}')
                relations.append(relation)

            writer.writerow([' '.join(wordLists[idx]), ' '.join(nums), ' '.join(relations)])

    with open('sentvec/group.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['head編號', 'index編號', 'head詞', 'index詞', 'Relation'])

        for idx, dep in enumerate(dependencies):
            for index, (head, relation) in enumerate(dep):
                words = wordLists[idx]
                headWord = words[head - 1] if head != 0 else 'ROOT'
                indexWord = words[index]
                writer.writerow([head, index + 1, headWord, indexWord, relation])

    with open('sentvec/dependencies.txt', 'w') as f:
        for dep in dependencies:
            nums = []
            for index, (head, relation) in enumerate(dep):
                nums.append(f'{head}->{index + 1}')

            f.write('<start> ' + ' '.join(nums) + ' <end>' + '\n')

def dep_skipgram(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(options, k_, v_)

    os.system(f'./fasttext skipgram -input sentvec/dependencies.txt -output {options.sest_result}/sest_skipgram -dim {options.sest_dim} -epoch {options.sest_epochs} -ws {options.sest_ws} -lrUpdateRate {options.sest_lr_update_rate} -thread {options.thread}')

def visualization(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(options, k_, v_)

    # Declaring two empty lists. One for words and one for their corresponding vector representation
    words = []
    vecs = []

    # Getting word and corresponding vector from each line of the model.vec file generated by fasttext
    with codecs.open('sentvec/skipgram.vec', 'r', 'utf-8') as f_in:
        vocabulary, wv = zip(*[line.strip().split(' ', 1) for line in f_in])

    # Populating the two lists. Need to convert vector values from string to numpy array
    for i in range(10,len(vocabulary)): # Usually skip first 10 words becuase they might be garbage values.
        words.append(vocabulary[i])
        x = wv[i]
        vecs.append(np.fromstring(x,dtype='float32',sep=' '))
        # np.fromstring takes string values and converts to float32 with space as a separator

    # Carrying out Singular Value Decomposition
    U, s, Vh = np.linalg.svd(vecs,full_matrices=False)

    # Plotting words and their vector representations
    plt.scatter(U[:, 0], U[:, 1])
    plt.savefig('sentvec/viz.jpg')

def run_early_fusion_lstm(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(options, k_, v_)

    device = torch.device('cuda' if torch.cuda.is_available() and options.cuda == True else 'cpu')


if __name__ == '__main__':
  fire.Fire()
