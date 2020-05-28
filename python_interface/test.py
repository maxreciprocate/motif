import pickle
import os

import numpy as np
from Genome import Genome
from CandidateMarkerFinder import CandidateMarkerFinder
from DataController import DataController
from MarkersSet import MarkersSet


def read_genomes(genome_list_filename):
    #genome_paths_list = []
    with open(genome_list_filename) as genome_list_file:
        genome_paths_list = [
            (
                os.path.join(
                    os.path.dirname(genome_list_filename),
                    genome_filename.strip()
                )
            )
            for genome_filename in genome_list_file.readlines()
            if genome_filename.strip()
        ]

    genome_list = [None for _ in range(len(genome_paths_list))]

    for i, genome_path in enumerate(genome_paths_list):
        with open(genome_path) as genome_file:
            print(str(i) + '. ' + genome_path + ' is read')
            yield Genome(genome_path, genome_file.read())


def test_algorithm(presence_matrix_test):
    markers_list = presence_matrix_test._markers

    candidate_marker_finder = CandidateMarkerFinder("test")
    candidate_marker_finder.marker_set = MarkersSet(markers_list)
    genomes_gen = read_genomes(os.environ['GENOMES_LIST_PATH'])
    genome_bundle_size = 1
    genomes = [None for _ in range(genome_bundle_size)]
    continue_ = True
    while continue_:
        try:
            for k in range(genome_bundle_size):
                genomes[k] = next(genomes_gen)
        except StopIteration:
            continue_ = False

        genomes[0] = Genome('data/bank/pseudo9909.fasta', open('data/bank/pseudo9909.fasta').read())
        candidate_marker_finder.data_controller = DataController("data_mode", genomes)

        candidate_marker_finder.run()
        presence_matrix = candidate_marker_finder.presence_matrix.presence_matrix

        for genome in presence_matrix_test._genomes.keys():
            if genome.name == '9909':
                print('genome index:', presence_matrix_test._genomes[genome])

        for i in range(len(presence_matrix[0])):
            if presence_matrix[0][i] != presence_matrix_test._presence_matrix[106][i] and presence_matrix[0][i] == 1:
                print('marker index', i)
                print(list(presence_matrix_test._markers.keys())[i]._sequence)
                print(presence_matrix[0][i], presence_matrix_test._presence_matrix[106][i])
                break
        np.testing.assert_array_equal(presence_matrix[0], presence_matrix_test._presence_matrix[106])
        break


def get_all_pickles(pickles_dir):
    return [
        os.path.join(pickles_dir, file) for file in os.listdir(pickles_dir)
        if (
                os.path.isfile(os.path.join(pickles_dir, file)) and
                os.path.splitext(os.path.join(pickles_dir, file))[1] == '.pickle'
        )
    ]


if __name__ == '__main__':
    pickles = get_all_pickles(os.environ['TESTS_DIR_PATH'])
    print('Pickles are ready')

    for pickle_filename in pickles:
        with open(pickle_filename, 'rb') as pickle_file:
            print(pickle_filename + ' testing started')
            presence_matrix_test = pickle.load(pickle_file)
            #print({i: genome.name for i, genome in enumerate(presence_matrix_test._genomes)})
            test_algorithm(presence_matrix_test)
            print(pickle_filename + ' testing finished')
        break
