#!/usr/bin/env pypy3
from ahocorapy.keywordtree import KeywordTree
import csv
import os
import sys
# from copy import deepcopy
import multiprocessing as mp
import time 

def search_markers(ar):
    kwtree, sourcefile, nmarkers = ar[0], ar[1], ar[2]
    with open(sourcefile) as file:
        source = file.readlines()[0]

    output = bytearray(nmarkers)
    for string, _ in kwtree.search_all(source):
        output[markersids[string]] = 0x01

    for idx in range(0, len(output)):
        output[idx] += ord('0')

    return output, sourcefile
    

if len(sys.argv) != 4:
    print(f"usage: {sys.argv[0]} <sources_file> <markers_file> <output_file> ")
    exit(1)

_, sources_file, markers_file, output_file = sys.argv
# n_threads = int(n_threads)
# sources_file should have relative filepaths to source files
prefix = "/".join(sources_file.split('/')[:-1])
with open(sources_file) as file:
    sourcefiles = [prefix + '/' + line.rstrip('\n') for line in file]
print(sourcefiles)
if len(sourcefiles) < 1:
    print("sources_file must have at least one source filepath")
    exit(1)


kwtree = KeywordTree(case_insensitive=True)
# lookup table for the relative index of each marker to the order they came in
markersids = {}

print("building")
with open(markers_file) as file:
    markers = [line for line in csv.reader(file)]

for idx, marker in markers:
    markersids[marker] = int(idx)
    kwtree.add(marker)

kwtree.finalize()
print("finished")

output_file = open(output_file, 'w')
p = mp.Pool(processes=4)
#p = mp.Pool(processes=int(os.environ["PYPY_NUM_THREADS"))
n= len(markers)
start = time.time()
for out, source in p.imap_unordered(search_markers, [(kwtree, s, n) for s in sourcefiles]):
    output_file.write(source.split('/')[-1] + " ")
    output_file.write(out.decode('utf-8'))
    output_file.write('\n')
print(time.time() - start)
