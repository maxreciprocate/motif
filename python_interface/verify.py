import jam_lib
import numpy as np

def verify(genome_fname, markers_fname, res_file_name):
  markers = []
  
  with open(genome_fname,"r") as f:
    genome = f.readline()
    
  with open(markers_fname, "r") as f:
    for line in f.readlines():
      marker = line.strip().split(",")[1]
      markers.append(marker)

  np_res = jam_lib.run(["pseudo88.fasta"], [genome], markers, 1)
  
  with open(res_file_name, "r") as f:
    genome = f.readline().strip()
    result = genome.split(" ")[1]
    res = np.array(np_res[0][1], dtype=np.int8)
    print(type(res))
    for i, ch in enumerate(result):
      if int(ch) == res[i]:
        continue
      else: 
        raise RuntimeError("Failed")
  print("Verified")
  return 0

if __name__ == "__main__":
    verify("data/bank/pseudo88.fasta", "data/8000markers.csv", "data/out_now.txt")
