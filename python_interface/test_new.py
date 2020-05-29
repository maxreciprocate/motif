import pickle
import os

import numpy as np
from Genome import Genome
from CandidateMarkerFinder import CandidateMarkerFinder
from DataController import DataController
from DataConnector import DataConnector
from MarkersSet import MarkersSet

path = "/home/myrddin/arabidopsis_full_genomes"

def test_algorithm(presence_matrix_test):
    markers_list = presence_matrix_test._markers

    candidate_marker_finder = CandidateMarkerFinder("test")
    candidate_marker_finder.marker_set = MarkersSet(markers_list)

    genome_list = [None for _ in range(len(presence_matrix_test._genomes))]

    for i, (genome, genome_index) in enumerate(presence_matrix_test._genomes.items()):
        genome_list[genome_index] = Genome(name=genome.name)
        genome_list[genome_index].add_string_data_connector(
            DataConnector(
                file=os.path.join(path, 'data/string', genome.name + '.pickle')
            )
        )

    candidate_marker_finder.data_controller = DataController("data_type", genome_list)

    candidate_marker_finder.run()

    np.testing.assert_array_equal(
        candidate_marker_finder._presence_matrix._presence_matrix,
        presence_matrix_test._presence_matrix
    )


def get_all_pickles(pickles_dir):
    return [
        os.path.join(pickles_dir, file) for file in os.listdir(pickles_dir)
        if (
                os.path.isfile(os.path.join(pickles_dir, file)) and
                os.path.splitext(os.path.join(pickles_dir, file))[1] == '.pickle'
        )
    ]


if __name__ == '__main__':
    # pickles = get_all_pickles(os.environ['TESTS_DIR_PATH'])
    pickles = ['../test_cases/presence_matrix_5.pickle']

    print('Pickles are ready')

    for pickle_filename in pickles:
        with open(pickle_filename, 'rb') as pickle_file:
            print(pickle_filename + ' testing started')
            presence_matrix_test = pickle.load(pickle_file)
            test_algorithm(presence_matrix_test)
            print(pickle_filename + ' testing finished')
        break

