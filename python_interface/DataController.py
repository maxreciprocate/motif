class DataController:
    def __init__(self, data_mode, genomes):
        self.data_mode = data_mode
        self.genomes = genomes

    def get_all_genomes(self):
        """
        :return: list of Genome instances
        """
        return self.genomes

