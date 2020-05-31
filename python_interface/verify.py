import jam_lib
import numpy as np
import sys
import time
import sys
import subprocess

def verify(genomefnsource, markers_fname, res_file_name):
  
  start = time.time()
  # subprocess.run(["./jam/jam", genomefnsource, markers_file, res_file_name], stdout=subprocess.PIPE)
  now = time.time()
  print("start reading")
  markers = []
  genomes = []
  genome_names = []

  with open(genomefnsource) as f:
    genome_names = ["data/" + i.strip() for i in f.readlines()]

  now = time.time()
  for genome_fname in genome_names:
    with open(genome_fname,"r") as f:
      genome = f.readline()
      genomes.append(genome)
      
  with open(markers_fname, "r") as f:
    for line in f.readlines():
      marker = line.strip().split(",")[1]
      markers.append(marker)

  matrix = np.zeros((len(genomes),len(markers)), dtype=np.int8)
  print(now - start)
  
  jam_lib.run(genomes, markers, matrix, 1, False)
  
  print(time.time() - now)
  print("runned")
  # print(matrix)
  with open(res_file_name, "r") as f:
    lines = f.readlines()
    for j in range(matrix.shape[0]):
      for line in lines:
        name, genome = line.strip().split(" ")
        if name == genome_names[j]:
          row = matrix[j]
          for i, ch in enumerate(genome):
            if int(ch) != row[i]:
              print("Failed for ", name)
              break
          print(name) 
    print("Verified")
  return 0

if __name__ == "__main__":
    # np.set_printoptions(threshold=np. inf)
    if len(sys.argv) != 4:
      print(f"usage: {sys.argv[0]} <sources_file> <markers_file> <output_file>")
      exit(1)

    _, sources_file, markers_file, output_file = sys.argv
    verify(sources_file, markers_file, output_file)

