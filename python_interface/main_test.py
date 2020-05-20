import numpy as np
from CandidateMarkerFinder import CandidateMarkerFinder
from DataController import DataController
from Genome import Genome
from Marker import Marker
from MarkersSet import MarkersSet
from PresenceMatrix import PresenceMatrix

if __name__ == "__main__":

    markers_list = [
        Marker(0, "AAA"),
        Marker(1, "CAC"),
        Marker(2, "TTGA")
    ]
    genomes_list = [
        Genome("test1", "AAACNNTTAC"),  # expect 100
        Genome("test2", "TTATTGACAT"),  # expect 001
        Genome("test3", "TTACACTAAA"),  # expect 110
    ]

    candidate_marker_finder = CandidateMarkerFinder("test")

    candidate_marker_finder.marker_set = MarkersSet(markers_list)

    candidate_marker_finder.data_controller = DataController("data_mode", genomes_list)

    a = PresenceMatrix()

    candidate_marker_finder.run()

    presence_matrix = candidate_marker_finder.presence_matrix.presence_matrix

    np.testing.assert_array_equal(presence_matrix[0], np.array([1, 0, 0], dtype=np.uint8))
    np.testing.assert_array_equal(presence_matrix[1], np.array([0, 0, 1], dtype=np.uint8))
    np.testing.assert_array_equal(presence_matrix[2], np.array([1, 1, 0], dtype=np.uint8))
