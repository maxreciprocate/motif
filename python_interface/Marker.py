class Marker(object):
    _VALID_NUCLEOTIDES = ["A", "C", "G", "T"]

    def __init__(self, _id, phenotype, sequence=None):
        if not type(_id) is int:
            raise ValueError("Id must be int. Current id = {}. Current id type "
                             "={}.".format(_id, type(_id)))
        self._id = _id
        self._sequence = None
        self._sequence_initialized = False
        if sequence is not None:
            self._sequence = sequence
            self._sequence_initialized = True
        self._phenotype = phenotype

    @property
    def sequence(self):
        return self._sequence