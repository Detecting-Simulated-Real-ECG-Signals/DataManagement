'''
Module provides Data Object which can be used to add missions to LocalDataManagement
'''
import numpy as np


class ImportData:
    '''
    Data format used to add mission to DataManagement
    '''

    def __init__(self, uid_4: str, label: str, data: np.ndarray) -> None:
        self.uid_4 = uid_4
        self.label = label
        self.data = data
