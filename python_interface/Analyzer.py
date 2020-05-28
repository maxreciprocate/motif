import numpy as np
import jam_lib as jam


class Analyzer:
    def run(self, genome_names, np_genomes_array, np_markers_array):
        #print(genome_names)
        #return np.array([])
	# output_matrix = np.empty(shape=(genome_names.size, 2), dtype=np.chararray)
        #print(np_markers_array)
        return jam.run(genome_names,
                np_genomes_array,
                np_markers_array,
                1
                )

if __name__ == "__main__":
    genome_name = ['fasta1','fasta2','fasta3']
    genome_data = ["ACTAACC", "ATTTTAA", "AAAAAA"]
    markers = ["AA", "ACT", "TTTTA"]
    # out = np.empty((3,2), dtype=np.float)
    # out = np.
    an = Analyzer()
    out = an.run(genome_name, genome_data, markers)
    print(out)
