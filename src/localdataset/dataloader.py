import re
from abc import ABC, abstractmethod
from ctypes import c_wchar
from multiprocessing import RawArray
from typing import Tuple

import numpy as np
import polars as pl
import pyarrow as pa
from uuid import UUID

from .dtypes import GID, LABEL, TAGS, DataPaths


class DataLoader(ABC):
    '''
    Dataloader blueprint
    '''

    @abstractmethod
    def load_mission_data_by_index(self, index: int) -> np.ndarray:
        '''
        Load perprocessed mission data as numpy array

        Input:
        * index <int>: Index of the mission\n

        Return: data <np.ndarray>
        '''
        pass

    @abstractmethod
    def load_mission_label_by_index(self, index: int) -> str:
        '''
        Load mission label from info.arrow file

        Input:
        * index <int>: Index of the mission\n

        Return: label <str>
        '''
        pass

    @abstractmethod
    def load_mission_tags_by_index(self, index: int) -> str:
        '''
        Load mission label from info.arrow file

        Input:
        * index <int>: Index of the mission\n

        Return: tags <str>
        '''
        pass

    @abstractmethod
    def get_index_by_gid(self, gid: UUID) -> int:
        '''
        Get index for gid

        Input:
        * gid <UUID>: Mission GID

        Return: index <int>
        '''
        pass

    @abstractmethod
    def get_gid_by_index(self, index: int) -> GID:
        '''
        Get gid for index

        Input:
        * index <int>: Index of the mission\n

        Return: gid <UUID>
        '''
        pass

    def load_mission_informations_by_id(self, index: int) -> Tuple[UUID, LABEL, TAGS]:
        '''
        Load mission informations from info.arrow file

        Input:
        * index <int>: Index of the mission\n

        Return: gid <UUID>, LABEL <str>, TAGs <List[int]>
        '''
        return self.get_gid_by_index(index), self.load_mission_label_by_index(index), self.load_mission_tags_by_index(index)


class MemoryMappedLoader(DataLoader):
    '''
    Load data via memory mapping 
    '''

    def __init__(self, paths: DataPaths) -> None:
        super().__init__()

        with pa.memory_map(str(paths.data), 'r') as source:
            self.__data = pa.ipc.open_file(source).read_all()["data"]

        with pa.memory_map(str(paths.label), 'r') as source:
            self.__label = pa.ipc.open_file(source).read_all()["label"]

        if paths.info is not None:
            with pa.memory_map(str(paths.info), 'r') as source:
                self.__info = pa.ipc.open_file(source).read_all()
        else:
            self.__info = None

        with pa.memory_map(str(paths.uid), 'r') as source:
            self.__uuid = pa.ipc.open_file(source).read_all()

    def load_mission_data_by_index(self, index: int) -> np.ndarray:
        if self.__info is not None:
            index = self.__info[0][index].as_py()
        return self.__data[index].values.to_numpy()

    def load_mission_label_by_index(self, index: int) -> str:
        if self.__info is not None:
            index = self.__info[0][index].as_py()
        return self.__label[index].as_py()

    def load_mission_tags_by_index(self, index: int) -> str:
        if self.__info is not None:
            return self.__info[1][index].as_py()
        return []

    def get_gid_by_index(self, index: int) -> GID:
        return self.__uuid['gid'][index].as_py()

    def get_index_by_gid(self, gid: UUID) -> int:
        return np.argmax(self.__uuid['gid'].to_numpy() == gid)


class SharedMemoryLoader(DataLoader):
    '''
    Load data into shared memory RawArrays
    '''

    def __init__(self, paths: DataPaths) -> None:
        super().__init__()

        # load gid_map
        self.__uuids = RawArray(c_wchar, "".join(
            pl.read_ipc(paths.uid)["gid"].to_list()))

        # load base_gids and tags
        if paths.info is not None:
            info = pl.read_ipc(paths.info)
            self.__info = RawArray("i", info.to_series(0).to_list())
            self.__tags = RawArray("i", [i[0]
                                   for i in info.to_series(1).to_list()])
        else:
            self.__info = None
            self.__tags = None

        # load data
        mission_data = pl.read_ipc(paths.data)
        for i, d in enumerate(mission_data.to_series(0).to_list()):
            setattr(self, str(i), RawArray("d", d))

        # load labels
        base_info = pl.read_ipc(paths.label)
        label_map = {"real_mission": 0, "test_mission": 1, "test_missions": 1}
        i = 1
        self.__reverse_label_map = RawArray(
            c_wchar, "real_missiontest_mission")
        self.__labels = RawArray("i", [label_map[i]
                                 for i in base_info.to_series(0).to_list()])

    def load_mission_data_by_index(self, index: int) -> np.ndarray:
        if self.__info is not None:
            index = self.__info[index]
        return np.frombuffer(getattr(self, str(index)))

    def load_mission_label_by_index(self, index: int) -> str:
        if self.__info is not None:
            index = self.__info[index]
        i = self.__labels[index]
        return self.__reverse_label_map[i*12:12+i*12] # labels are all 12 char long -> use the length get the gid by index

    def load_mission_tags_by_index(self, index: int) -> str:
        if self.__tags is not None:
            return [self.__tags[index]]
        return []

    def get_gid_by_index(self, index: int) -> GID:
        
        return self.__uuids[index*36:36+index*36]   # uuids are all 36 char long -> use the length get the gid by index

    def get_index_by_gid(self, gid: UUID) -> int:
        return int(self.__uuids[:].index(gid) / 36) # uuids are all 36 char long -> use the length get the gid by index
