class Genome:
    def __init__(self, name):
        self.name = name
        self.data_connector = None

    def add_string_data_connector(self, data_connector):
        self.data_connector = data_connector

    def get_genome_string_data(self):
        """
        Returns:
            string: whole genome sequence
            dict:   some metadata
        """
        return self.data_connector.get_data()

    def get_genome_numeric_data(self):
        """
        Returns:
            numpy.array:    shape=(4,genome_length), dtype=int8, contains whole genome data
            dict:           some metadata
        """
        return
