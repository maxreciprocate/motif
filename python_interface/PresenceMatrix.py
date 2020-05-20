import os

import numpy as np


class PresenceMatrix:
    def __init__(self):
        self._presence_matrix = None
        self._initialized = False
        self._markers = None
        self._genomes = None

    def initialize(self, genomes_list, marker_number):
        """
        Initializes all internal data structures for a PresenceMatrix object. After
        initialization it is not possible to add any genomes to the matrix or change
        the number of markers.

        Args:
            genomes_list:   list of all genomes that will be used in a matrix
            marker_number:  total number of markers used in a matrix
        """
        if self.initialized:
            raise ValueError("presence matrix already initialized")
        if len(genomes_list) == 0:
            raise ValueError("empty genome_list")
        self._presence_matrix = np.zeros((len(genomes_list), marker_number), dtype='int8')
        self._genomes = {}
        for i, genome in enumerate(genomes_list):
            if genome in self._genomes:
                raise (ValueError("genomes in list not unique: {}".format(genome.name)))
            else:
                self._genomes[genome] = i
        self._markers = {}
        self._initialized = True

    def reset(self):
        """
        Clear state of the matrix.
        """
        if not self._initialized:
            return
        self._initialized = False
        self.initialize(list(self._genomes.keys), len(self._markers.keys()))

    def add_position(self, marker_occurence):
        """
        Args:
        marker_occurence: MarkerOccurence object with pair Genome-Marker to add to matrix
        """
        if not self._initialized:
            raise ValueError("presence matrix not initialized")
        self._markers[marker_occurence.marker] = marker_occurence.marker.id
        column_id = self._markers[marker_occurence.marker]
        if marker_occurence.genome not in self._genomes:
            raise (ValueError("invalid genome: {}".format(marker_occurence.genome.name)))
        row_id = self._genomes[marker_occurence.genome]
        self._presence_matrix[row_id, column_id] = 1

    def add_all_positions_for_genome(self, genome, all_marker_positions):
        """
        Add all marker positions for a specified genome.
        Args:
            genome:                 Genome
            all_marker_positions:   np.array, row vector of zeros and ones that encodes all
                                    occurrences of markers in a genome.
        Warning: Requires that marker with marker.id = i corresponds to column i.
        This method was created for performance efficiency and needs refactoring.
        """
        if genome not in self._genomes:
            raise (ValueError("invalid genome: {}".format(genome.name)))
        row_id = self._genomes[genome]
        self._presence_matrix[row_id] = all_marker_positions

    def get_marker_column(self, marker):
        return marker.id

    @property
    def marker_number(self):
        return self._presence_matrix.shape[1]

    @property
    def initialized(self):
        return self._initialized

    @property
    def genomes(self):
        return list(sorted(self._genomes, key=self._genomes.get))

    @property
    def markers(self):
        return list(sorted(self._markers, key=self._markers.get))

    @property
    def presence_matrix(self):
        return self._presence_matrix

    def save(self, path):
        """
        Pickle is unable to serialize matrices of size approximately (1.000, 10.000.000).
        This method stores data separately from the object.
        """
        pass
        # os.mkdir(path)
        # hf = h5py.File(os.path.join(path,"matrix.h5"), 'w')
        # hf.create_dataset('presence_matrix', data=self._presence_matrix) hf.close()
        # presence_matrix_temp = self._presence_matrix
        # self._presence_matrix = None
        # self._presence_matrix_path = os.path.join(path,"matrix.h5")
        # pickle.dump(self, open(os.path.join(path,"object.pickle"), 'wb'))
        # self._presence_matrix = presence_matrix_temp

    def __setstate__(self, newstate):
        pass
        # if "_presence_matrix_path" in newstate:
        #     print(newstate["_presence_matrix_path"])
        # hf = h5py.File(newstate["_presence_matrix_path"], 'r') matrix = hf.get('presence_matrix') newstate["_presence_matrix"] = np.array(matrix)
        # print("Loaded", newstate["_presence_matrix"].shape)
        # print(np.sum(newstate["_presence_matrix"])) self.__dict__.update(newstate)
