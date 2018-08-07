from tqdm import tqdm
import csv

def gen(myDir):
    words = []
    
    with open(f'{myDir}/education/education_simplified.csv') as f:
        for row in tqdm(csv.reader(f)):
            if row[2] != '': words.append(row[2])

    with open(f'{myDir}/yunliu/yunliu_simplified.txt') as f:
        for row in tqdm(f.readlines()):
            if row != '\n': words.append(row.rstrip('\n'))

    with open(f'{myDir}/dict.txt', 'a') as f:
        for word in words:
            f.write(f'{word}\n')