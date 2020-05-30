import numpy as np
import jam_lib as jam


class Analyzer:
    def run(self, np_genomes_array, np_markers_array, output_matrix):
        return jam.run(
                np_genomes_array,
                np_markers_array,
                output_matrix,
                0,
                False
        )

if __name__ == "__main__":
    genome_data = ["ACTAACC", "ATTTTAA", "AAAAAA"]
    markers = ["AA", "ACT", "TTTTA"]
    out = np.zeros((3,3), dtype=np.int8)
    an = Analyzer()
    an.run(genome_data, markers, out)
    print(out)
