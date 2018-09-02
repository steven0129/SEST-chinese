#coding=utf8
import os
import torch
import fire
import ltp
import myDict
import csv
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
        tmpSents = list(map(lambda x: ' '.join(x), wordLists))
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
            for head, relation in dep:
                nums.append(f'{head}->{index + 1}')
            
            f.write(' '.join(nums) + '\n')

def preprocess(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(options, k_, v_)

    heads = {}
    indexes = {}
    relations = {}

    with open(f'{DATA_DIR}/sky_dragon_relations.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if(heads.get(row[0]) == None):
                heads[row[0]] = []
                indexes[row[0]] = []
                relations[row[0]] = []

            heads[row[0]].append(row[1])
            indexes[row[0]].append(row[2])
            relations[row[0]].append(row[3])

    with open(f'{DATA_DIR}/preprocess.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for idx, (key, vals) in enumerate(relations.items()):
            print(f'{idx}, {key}')
            writer.writerow([idx, key])

def run_rnn(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(options, k_, v_)

    device = torch.device('cuda' if torch.cuda.is_available() and options.cuda == True else 'cpu')
    

if __name__ == '__main__':
  fire.Fire()
