import csv
from tqdm import tqdm

words = []
with open('education/education_simplified.csv') as f:
    for row in tqdm(csv.reader(f)):
        if row[2] != '': words.append(row[2])

with open('yunliu/yunliu_simplified.txt') as f:
    for row in tqdm(f.readlines()):
        if row != '\n': words.append(row.rstrip('\n'))

with open('dict.txt', 'a') as f:
    for word in words:
        f.write(f'{word}\n')
