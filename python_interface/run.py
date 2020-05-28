#!/usr/bin/env pypy3

import csv
import sys
import jam_lib
import numpy as np
import time

if len(sys.argv) != 4:
    print(f"usage: {sys.argv[0]} <sources_file> <markers_file> <output_file>")
    exit(1)

_, sources_file, markers_file, output_file = sys.argv

prefix = "/".join(sources_file.split('/')[:-1])
with open(sources_file) as file:
    sourcefiles = [prefix + '/' + line.rstrip('\n') for line in file]

if len(sourcefiles) < 1:
    print("sources_file must have at least one source filepath")
    exit(1)


print("building")
with open(markers_file) as file:
    markers = [line[1] for line in csv.reader(file)]

genomes, genome_names = [], []
for sourcefile in sourcefiles:
  with open(sourcefile, "r") as f:
    genomes.append(f.readline())
    genome_names.append(sourcefile.split("/")[-1])

print(genome_names)
start = time.time()
output = jam_lib.run([genome_names[0]], [genomes[0]], markers, 2)
print(output)
with open(output_file, "w") as f:
  for res in np.nditer(output):
    f.write(res[0] + " ")
    f.write(res[1])
    f.write('\n')
print(time.time() - start)