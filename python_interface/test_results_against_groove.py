import numpy as np
import csv
import sys
import time
sys.path.append("python_interface")

from python_interface.run import run

sources_file = "data/10genomes.txt"
markers_file = "data/8000markers.csv"

with open(sources_file) as file:
  genome_d = {name[5:-1]: index for index, name in enumerate(file.readlines())}

with open(markers_file) as file:
  markers_num = len(list(csv.reader(file)))


f = open("test/data/result_1000genomes_markers.txt")

result_groove = np.empty((len(genome_d), markers_num), dtype=np.int8)

for i, line in enumerate(f.readlines()):
  name, data = line.split(' ')

  result_groove[genome_d[name]] = np.fromstring(data[:-1], np.int8) - ord('0')

  print(i)

f.close()

start = time.time()

result_jem_lib = run(
  sources_file, markers_file, ""
)

print("Total time: {}".format(time.time() - start))


np.testing.assert_array_equal(
  result_groove,
  result_jem_lib
)

print("All results agreed")
