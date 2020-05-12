import time
import os, psutil
from groove import read_markers, create_automaton, MARKERS_DATA, match, AUTOMATON
import numpy as np

def search_m():
    search("../motif/data/1genomes.txt", "../data/800000markers.csv", "output.txt")

if __name__ == "__main__":
    pid = os.getpid()
    ps = psutil.Process(pid)
    start = time.time()
    m_data = MARKERS_DATA()
    # at = AUTOMATON()
    print("Reading Markers ...")
    read_markers(m_data, "../data/800000markers.csv")
    
    print("Consumed memory:  ", ps.memory_info().rss / 10**6)
    
    print("Creating Automaton ...")
    # at = AUTOMATON()
    at = create_automaton(m_data)
    # print(at)
    print("Consumed memory:  ", ps.memory_info().rss / 10**6)

    del m_data;
    print("Consumed memory:  ", ps.memory_info().rss / 10**6)
    
    arr = np.zeros(len(at), dtype=np.int8)
    
    print("Reading Genome ...")
    with open("../data/bank/pseudo10010.fasta", "r") as f:
      gen = f.read()
    
    print("Consumed memory:  ", ps.memory_info().rss / 10**6)
    print(arr)
    print("Searching ...")

    arr = match(gen, at, arr)

    print("Consumed memory:  ", ps.memory_info().rss / 10**6)
    print(time.time() - start)
    print(arr)
    
