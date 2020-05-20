class Genome:
    def __init__(self, name, data):
        self.name = name
        self.data = data

    def get_genome_string_data(self):
        """
        Returns:
            string: whole genome sequence
            dict:   some metadata
        """
        return self.data

    def get_genome_numeric_data(self):
        """
        Returns:
            numpy.array:    shape=(4,genome_length), dtype=float32, contains whole genome data
            dict:           some metadata
        """
        return
