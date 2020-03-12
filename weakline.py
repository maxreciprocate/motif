from ahocorapy.keywordtree import KeywordTree
import csv
from time import time
import sys

if len(sys.argv) < 4:
    print(f"usage: {sys.argv[0]} <source_file> <markers_file> <output_file>")
    exit(1)

source_file = sys.argv[1]
markers_file = sys.argv[2]
output_file = sys.argv[3]

start = time()
with open(source_file) as file:
    source = file.read()

kwtree = KeywordTree(case_insensitive=True)
# lookup table for the relative index of each marker to the order they came in
markersids = {}

with open(markers_file) as file:
    markers = [line for line in csv.reader(file)]

for idx, marker in markers:
    markersids[marker] = int(idx)
    kwtree.add(marker)

kwtree.finalize()

output = bytearray(len(markers))
for string, _ in kwtree.search_all(source):
    output[markersids[string]] = 0x01

duration = time() - start

print(f"timing: {duration:.4f}s")

for idx in range(0, len(output)):
    output[idx] += 48

with open(output_file, 'w+') as file:
    file.write(output.decode('utf-8'))
