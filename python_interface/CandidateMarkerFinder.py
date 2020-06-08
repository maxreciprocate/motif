import numpy as np
from PresenceMatrix import PresenceMatrix
#from Analyzer import Analyzer
import jam_lib
import time

class ExtractionStage:
    def __init__(self, name):
        pass

    def save(self, file_name):
        pass


class StageAttributeAccessError:
    def __init__(self, name):
        pass

class CandidateMarkerFinder(ExtractionStage):
    def __init__(self, name):
        ExtractionStage.__init__(self, name)
        self._marker_set = None
        self._data_controller = None
        self._chromosome = None
        self._data_mode = None
        self._presence_matrix = None
        self.completed = False
        self.clear_state()

    @property
    def marker_set(self):
        if self._marker_set is None:
            raise StageAttributeAccessError("Marker set not initialized")
        return self._marker_set

    @marker_set.setter
    def marker_set(self, value):
        self._marker_set = value
        self.clear_state()

    @property
    def data_controller(self):
        if self._data_controller is None:
            raise ValueError("data_controller not initialized")
        return self._data_controller

    @data_controller.setter
    def data_controller(self, value):
        self._data_controller = value
        self._data_mode = self._data_controller.data_mode
        self.clear_state()

    @property
    def data_mode(self):
        if self._data_mode is None:
            raise ValueError("data_mode not initialized")
        return self._data_mode

    @property
    def chromosome(self):
        if self._chromosome is None:
            raise AttributeError("chromosome not initialized")
        return self._chromosome

    @chromosome.setter
    def chromosome(self, value):
        if self._chromosome is not None:
            raise AttributeError("chromosome already initialized")
        self._chromosome = value

    @property
    def presence_matrix(self):
        if not self.completed:
            raise StageAttributeAccessError("Candidate marker search not completed")
        return self._presence_matrix

    def clear_state(self):
        if self._presence_matrix is not None:
            self._presence_matrix.reset()
        self.completed = False

    def save(self, file_name):
        super().save(file_name)
        if self._data_controller is not None:
            self.data_controller.reset()

    def run(self):
        """
        For an implementation of an interface, call to this method should execute an algorithm of marker search.
        """
        # This part checks if all required fields are initialized
        # self.data_mode
        # if self.data_mode == "chromosome":
        #     self.chromosome

        markers = [marker._sequence for marker in self.marker_set.get_marker_list()]
        genomes = self.data_controller.get_all_genomes()

        counter = 0
        block_size = 100

        block_genomes = [None for _ in range(block_size)]

        self._presence_matrix = PresenceMatrix()
        self._presence_matrix.initialize(genomes, len(markers))

        analyzer = jam_lib.Analyzer()

        selected_gpu_devices = [1, 1, 1, 1]

        analyzer.build(
          markers,
          selected_gpu_devices
        )

        while counter < len(genomes):
            # fill block with data
            block_counter = 0
            max_genome_length = 0
            read_start = time.time()
            while block_counter < block_size and counter < len(genomes):
                block_genomes[block_counter] = genomes[counter].get_genome_string_data()
                max_genome_length = max(max_genome_length, len(block_genomes[block_counter]))
                block_counter += 1
                counter += 1

            # slice block if needed
            if block_counter < block_size - 1:
                block_genomes = block_genomes[:block_counter]

            read_end = time.time()

            matrix_result = np.zeros((len(block_genomes), len(markers)), dtype=np.int8)

            isGenomeInNumpyForm = False

            analyzer.run(
              block_genomes,
              max_genome_length,
              matrix_result,
              isGenomeInNumpyForm
            )

            for i, genome_result in enumerate(matrix_result):
                self._presence_matrix.add_all_positions_for_genome(genomes[counter - block_counter + i], genome_result)

            print("Reading block time: {}, analyzing + saving: {}".format(read_end - read_start, time.time() - read_end))

        analyzer.clear()

        self.completed = True

