import os
import re
import sys
import csv

folder = sys.argv[1]
folder_name = os.path.split(folder)[1]
output_file = os.path.join(folder, 'metrics_' + folder_name + '.csv')
files = list(filter(lambda x: x.startswith('metric') and x.endswith('.txt'), os.listdir(folder)))
files.sort(key=lambda x: int(re.findall(r'[0-9]+', x)[0]))

keys = set()
data = []

for file in files:
    f = open(os.path.join(folder, file), 'r')
    content = f.read()
    groups = re.findall(r'(Bleu_[1-4]|METEOR|ROUGE_L|CIDEr|SPICE)(:)([0-9.]+)', content)
    rowdata = {}
    for group in groups:
        keys.add(group[0])
        rowdata[group[0]] = group[2]
        rowdata['_'] = file
    data.append(rowdata)

keys = list(keys)
keys.sort()
keys = ['_'] + keys

f_csv = open(output_file, 'w')
writer = csv.writer(f_csv)

writer.writerow(keys)
for rowdata in data:
    content = [rowdata[k] for k in keys]
    writer.writerow(content)
f_csv.close()

