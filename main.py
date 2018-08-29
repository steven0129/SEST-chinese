#coding=utf8
import os
import fire
import ltp
import myDict
from tqdm import tqdm
from config import Env

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
    
    print(wordLists[0])
    print(S.parse(wordLists[0]))

if __name__ == '__main__':
  fire.Fire()
