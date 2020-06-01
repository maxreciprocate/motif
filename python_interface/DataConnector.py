import pickle


class DataConnector:
    def __init__(self, file):
        self.file = file

    def get_data(self):
        '''
        Read data from a file and reformat in to a specific format Returns: either string or numpy.array, dtype=float32
        '''
        f = open(self.file)
        buff = f.read()
        f.close()
        return buff

#i = DataConnector('./data/bank/pseudo88.fasta')
#print(i.get_data())
