from ahocorapy.keywordtree import KeywordTree
import csv
from time import time
import sys

if len(sys.argv) < 2:
    print(f"usage: {sys.argv[0]} <number_of_markers_to_add>")
    exit(1)

nmarkers = int(sys.argv[1])
print(f"collecting {nmarkers} markers")

with open("data/thaliana.fna") as file:
    source = file.read()

kwtree = KeywordTree(case_insensitive=True)
with open("data/markers.csv") as file:
    for line in csv.reader(file):
        nmarkers -= 1
        if nmarkers <= 0: break

        kwtree.add(line[1])

print("finished adding")

kwtree.finalize()
print("done with building")

def search(tree, source):
    count = 0
    for string, _ in tree.search_all(source):
        count += 1

    return count

start = time()
count = search(kwtree, source)
duration = time() - start

print(f"took {duration} seconds with #{count} matches")
