import numpy as np
from PresenceMatrix import PresenceMatrix
from Analyzer import Analyzer


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
        block_genomes_names = [None for _ in range(block_size)]

        self._presence_matrix = PresenceMatrix()
        self._presence_matrix.initialize(genomes, len(markers))

        while counter < len(genomes):
            # fill blocks with data
            block_counter = 0
            while block_counter < block_size and counter < len(genomes):
                block_genomes[block_counter] = genomes[block_counter].get_genome_string_data()
                block_genomes_names[block_counter] = genomes[block_counter].name
                block_counter += 1
                counter += 1

            # slice blocks if needed
            if block_counter < block_size - 1:
                block_genomes = block_genomes[:block_counter]
                block_genomes_names = block_genomes_names[:block_counter]

            matrix = Analyzer().run(
                block_genomes_names,
                block_genomes,
                markers
            )
            for (genome_name, genome_result) in matrix:
                for genome in genomes:
                    if genome.name == genome_name:
                        self._presence_matrix.add_all_positions_for_genome(genome, genome_result)

        self.completed = True

