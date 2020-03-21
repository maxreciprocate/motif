#!/usr/bin/env pypy3
from ahocorapy.keywordtree import KeywordTree
import csv
import os
import sys
# from copy import deepcopy
import multiprocessing as mp
import concurrent.futures
import time
import ctypes

kwtree = KeywordTree(case_insensitive=False)

def search_markers(sourcefile_arr, output_file, lock, nmarkers):
    for sourcefile in sourcefile_arr:
        sourcefile = sourcefile.value.decode('utf-8')
        print("reading sourcefile {}".format(sourcefile))

        lock.acquire()
        try:
            with open(sourcefile, 'r') as file:
                source = file.readlines()[0]
        except MemoryError:
            print("Got an MemoryError working with {}".format(sourcefile))
        finally:        
            lock.release()

        print("searching in {}".format(sourcefile))
        output = bytearray(nmarkers)
        for string, _ in kwtree.search_all(source):
            for mark_id in markersids[string]:
                output[mark_id - 1] = 0x01

        for idx in range(0, len(output)):
            output[idx] += ord('0')
        
        lock.acquire()
        with open(output_file.value.decode('utf-8'), 'a') as f:
            f.write(sourcefile.split('/')[-1] + " ")
            f.write(output.decode('utf-8'))
            f.write('\n')
        lock.release()    

        print("Done with {}".format(sourcefile))
        del output
        del source

    
if len(sys.argv) != 4:
    print(f"usage: {sys.argv[0]} <sources_file> <markers_file> <output_file> ")
    exit(1)

_, sources_file, markers_file, output_file = sys.argv

prefix = "/".join(sources_file.split('/')[:-1])
with open(sources_file) as file:
    sourcefiles = [prefix + '/' + line.rstrip('\n') for line in file]
if len(sourcefiles) < 1:
    print("sources_file must have at least one source filepath")
    exit(1)


markersids = {}

print("building")
with open(markers_file) as file:
    markers = [line for line in csv.reader(file)]

for idx, marker in markers:
    if markersids.get(marker, None):
        markersids[marker].append(int(idx))
    else:
        markersids[marker] = [int(idx)]
    kwtree.add(marker)

kwtree.finalize()
print("finished")

n = len(markers)

# clear output file
f = open(output_file, 'w')
f.close()

# encode output filename for ctypes value
output_file = output_file.encode('utf-8')
out_file = mp.Value(ctypes.c_char_p, output_file)

processes = []
l = mp.Lock()

n_threads = int(os.environ.get('PYPY_NUM_THREADS', 1))
files_per_thread, files_left = len(sourcefiles) // n_threads, len(sourcefiles) % n_threads

for pr in range(n_threads):
    n_files = files_per_thread + 1 if files_left else files_per_thread
    files_left = max(files_left-1, 0)
    s_files = []

    for f in range(n_files):
        try:
            f_name = sourcefiles.pop()
        except:
            break
        s_files.append(mp.Value(ctypes.c_char_p, f_name.encode('utf-8')))

    new_process = mp.Process(target=search_markers, args=(s_files, out_file, l, n))
    processes.append(new_process)

for pr in processes: pr.start()
for pr in processes: pr.join() 