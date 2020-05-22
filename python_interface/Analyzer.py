import numpy as np


class Analyzer:
    def run(self, np_genomes_array, np_markers_array):
        return np.zeros((len(np_genomes_array), len(np_markers_array)), dtype='int8')