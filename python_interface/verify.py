import jam_lib
import numpy as np
import time
import subprocess

def verify(genomefnsource, markers_fname, res_file_name):
  
  start = time.time()
  subprocess.run(["./jam/jam", "data/50genomes.txt", "data/800000markers.csv", "data/out_now.txt"], stdout=subprocess.PIPE)
  now = time.time()
  print(now - start)
  markers = []
  genomes = []
  genome_names = []

  with open(genomefnsource) as f:
    genome_names = ["data/" + i.strip() for i in f.readlines()]

  for genome_fname in genome_names:
    with open(genome_fname,"r") as f:
      genome = f.readline()
      genomes.append(genome)
      
  with open(markers_fname, "r") as f:
    for line in f.readlines():
      marker = line.strip().split(",")[1]
      markers.append(marker)

  matrix = np.zeros((len(genomes),len(markers)), dtype=np.int8)
  jam_lib.run(genomes, markers, matrix, 1)
  print(time.time() - now)
  print("runned")

  with open(res_file_name, "r") as f:
    j = 0
    genome = f.readline().strip()
    
    while genome:
      name, result = genome.split(" ")
      res = matrix[j]

      for i, ch in enumerate(result):
        if int(ch) == res[i]:
          continue
        else: 
          raise RuntimeError("Failed")

      print("Verified ", name)
      j += 1
      genome = f.readline().strip()

  return 0

if __name__ == "__main__":
    np.set_printoptions(threshold=np. inf)
    verify("data/50genomes.txt", 
    "data/800000markers.csv", "data/out_now.txt")

