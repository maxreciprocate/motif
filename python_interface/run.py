import numpy as np
import time

from CandidateMarkerFinder import CandidateMarkerFinder
from MarkersSet import MarkersSet
from Marker import Marker
from Genome import Genome
from DataConnector import DataConnector
from DataController import DataController

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

genome_names = [sourcefile.split("/")[-1] for sourcefile in sourcefiles]

candidate_marker_finder = CandidateMarkerFinder("test")
candidate_marker_finder.marker_set = MarkersSet(
    [Marker(i, 'phenotype', sequence=marker) for i, marker in enumerate(markers)]
)

genome_list = [None for _ in range(len(genome_names))]

for i, genome_name in enumerate(genome_names):
    genome_list[i] = Genome(name=genome_name)
    genome_list[i].add_string_data_connector(
        DataConnector(file=sourcefiles[i])
    )

candidate_marker_finder.data_controller = DataController("data_type", genome_list)

candidate_marker_finder.run()

output = candidate_marker_finder._presence_matrix._presence_matrix

with open(output_file, "w") as f:
  for i, res in enumerate(output):
    f.write(genome_names[i])
    f.write(' ')
    f.write(''.join(str(chr) for chr in res))
    f.write('\n')

