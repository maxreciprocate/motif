import jam_lib
import numpy as np
import sys
import time
import sys
import subprocess
import run


def verify(genomefnsource, markers_fname, res_file_name):
  start_jam = time.time()
  subprocess.run(["./jam/jam", genomefnsource, markers_fname, res_file_name], stdout=subprocess.PIPE)
  print("Jam time:" , time.time() - start_jam)

  start_jam_lib = time.time()
  matrix = run.run(genomefnsource, markers_fname, res_file_name)
  print("Jam_lib time: ", time.time() - start_jam_lib)

  with open(genomefnsource) as f:
    genome_names = ["data/" + line.strip() for line in f.readlines()]

  with open(res_file_name, "r") as f:
    lines = [line.strip().split(" ") for line in f.readlines()]
    for j in range(matrix.shape[0]):
      for line in lines:
        name, genome = line
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

    verify(*sys.argv[1:])

