import numpy as np
import jam_lib as jam


class Analyzer:
<<<<<<< Updated upstream
    def run(self, np_genomes_array, max_genome_length, np_markers_array, n_devices, is_numpy):
        output = np.zeros((len(np_genomes_array), len(np_markers_array)), dtype=np.int8)
        jam.run(np_genomes_array, max_genome_length, np_markers_array, output, n_devices, is_numpy)
=======
    def run(self, np_genomes_array, np_markers_array, device_arr, is_numpy):
        output = np.zeros((len(np_genomes_array), len(np_markers_array)), dtype=np.int8)
        jam.run(np_genomes_array, np_markers_array, output, device_arr, is_numpy)
>>>>>>> Stashed changes
        return output

if __name__ == "__main__":
    genome_data = ["ACTAACC", "ATTTTAA", "AAAAAA"]
    markers = ["AA", "ACT", "TTTTA"]
    out = np.zeros((3,3), dtype=np.int8)
    an = Analyzer()
    an.run(genome_data, markers, out)
    print(out)
