import marshal
import os
import uuid
from abc import ABC, abstractmethod
from itertools import chain
from os import makedirs
from pathlib import Path
from random import shuffle
from shutil import copy2 as copy
from shutil import rmtree as rm_dirs
from types import FunctionType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from DataManagement.src.localdataset.import_data import ImportData
import numpy as np
import polars as pl
import pyarrow as pa
import yaml
from tqdm import tqdm

from .dataloader import DataLoader, MemoryMappedLoader
from .dtypes import GID, INDICES, LABEL, TAGS, DataPaths
from .fourier import fourier_transformation_normalized_avg_histogram

PA_SCHEMA = pa.schema([
    pa.field('nums', pa.float64())
])


class Mission:
    '''
    Mission contains the measurements, gid, label and additional tags for an analyse mission or preprocessed data element.
    '''

    def __init__(self, index: int, retrievel_class: DataLoader, recreation_method: Callable, recreation_args: list) -> None:
        self.__retrievel_class: DataWrapper = retrievel_class

        self.__recreation_method: Callable = recreation_method
        self.__recreation_args: list = recreation_args

        self.__index: int = index

        # load base informations

        gid, label, tags = self.__retrievel_class.load_mission_informations_by_id(
            self.__index)

        self.gid: GID = gid
        self.label: LABEL = label
        self.tags: TAGS = tags

    @property
    def data(self) -> np.ndarray:
        return self.__recreation_method(self.__retrievel_class.load_mission_data_by_index(self.__index), self.tags, *self.__recreation_args)


class AbstractDataClass(ABC):
    def __init__(self, retrievel_class: DataLoader, recreation_method: Callable, recreation_args: list, indices: list) -> None:
        self.retrievel_class = retrievel_class
        self.recreation_method = recreation_method
        self.recreation_args = recreation_args
        self.indices = indices

        # position
        self.__position = 0

    @abstractmethod
    def get_by_index(self, index: int) -> object:
        '''
        get data object at index.
        '''
        pass

    def __iter__(self):
        return self

    def __next__(self) -> object:
        if self.__position == len(self.indices):
            self.__position = 0
            raise StopIteration()

        self.__position += 1
        return self.get_by_index(self.__position - 1)

    def __len__(self) -> int:
        return len(self.indices)


class Data(AbstractDataClass):
    '''
    Iterable Data object returns Missions.
    '''

    def __init__(self, retrievel_class: DataLoader, recreation_method: Callable, recreation_args: list, indices: list) -> None:
        super().__init__(retrievel_class, recreation_method, recreation_args, indices)

    def get_by_index(self, index: int):
        return Mission(index, self.retrievel_class, self.recreation_method, self.recreation_args)


class lightWeightData(AbstractDataClass):
    def __init__(self, retrievel_class: DataLoader, recreation_method: Callable, recreation_args: list, indices: list) -> None:
        super().__init__(retrievel_class, recreation_method, recreation_args, indices)

    def get_by_index(self, id):
        return self.recreation_method(self.retrievel_class.load_mission_data_by_index(id), self.retrievel_class.load_mission_tags_by_index(id), *self.recreation_args), self.retrievel_class.load_mission_label_by_index(id)


class DataWrapper:
    def __init__(self, data: Dict[str, INDICES], retrievel_class: DataLoader, recreation_method: Callable, recreation_args: list, lightweight=False) -> None:

        if lightweight:
            object: AbstractDataClass = lightWeightData
        else:
            object: AbstractDataClass = Data

        for key in data.keys():
            setattr(self, key, object(retrievel_class,
                    recreation_method, recreation_args, data[key]))


def relevant_mission(data: np.ndarray, threshold=1) -> bool:
    avg_fft = fourier_transformation_normalized_avg_histogram(data, 300)
    avg_fft_normalized = avg_fft / avg_fft.sum()
    return avg_fft_normalized[1:].sum() >= threshold


def split_function(data: List[Any], train_val_test_ration: List[int]) -> Dict[str, Tuple[str, str]]:
    if len(train_val_test_ration) != 3:
        raise AssertionError(
            "train_val_test must contain one value for each set!")
    if sum(train_val_test_ration) != 1:
        raise AssertionError("train_val_test must sum up to 100% !")

    shuffle(data)  # random shuffle

    # split dataset
    return {"train": data[:round(len(data)*0.6)], "validation": data[round(len(data)*0.6):round(len(data)*0.7)], "test": data[round(len(data)*0.7):]}


class LocalDataset:
    '''
    Dataset class to download missions of CAED.
    '''
    __local_path: Path  # Base path

    __preprocessing_path: Path  # preprocessing folder containing preprocessing infos
    # contains dirs with the different preprocessing infos
    #
    # this dir can have two structures:
    #
    # |- <preprocessing-name>
    # |  |- info.arrow                  contains label, relevant and split
    # |  |- index_gid_map.arrow         to map gid to index
    # |  |- mission_data.arrow          contains the recorded mission data for each index
    # |
    #
    # And:
    #
    # |- <preprocessing-name>
    # |  |- info.arrow                  contains label, relevant and split
    # |  |- index_gid_map.arrow         to map gid to index
    # |  |- reconstruction_infos.dat    contains the recorded mission data for each index
    # |
    # ...

    __raw_mission_path: Path
    # contains dirs with the different dataset versions
    # |- V1
    # |  |- info.arrow                  contains label, relevant and split
    # |  |- index_gid_map.arrow         to map gid to index
    # |  |- mission_data.arrow          contains the recorded mission data for each index
    # |
    # |- V2
    # ...

    __tmp_folder: Path

    def __init__(self, local_path: Path = None) -> None:
        # load old local file path
        data_path = Path(__file__).parent / "data_path.yaml"
        if local_path is None:
            if data_path.exists():
                with open(data_path, 'r') as file:
                    local_path = Path(yaml.safe_load(file)["DATA_PATH"])
            else:
                AssertionError(
                    "No data-storage path yet. You must define one on first initialization!")
        else:
            with open(data_path, 'w') as file:
                yaml.safe_dump({"DATA_PATH": str(local_path)}, file)

        # setup paths
        self.__local_path = Path(local_path)  # base path

        self.__raw_mission_path = self.__local_path / "raw"  # for raw data
        makedirs(self.__raw_mission_path, exist_ok=True)

        self.__preprocessing_path = self.__local_path / \
            "preprocessing"  # for preprocessed data
        makedirs(self.__preprocessing_path, exist_ok=True)

        self.__tmp_folder = Path("/tmp/localdatamanagementtool/")
        makedirs(self.__tmp_folder, exist_ok=True)

    #################################################################################################################
    #                                                                                                               #
    #                                          retrive mission data                                                 #
    #                                                                                                               #
    #################################################################################################################

    def __read_reconstruction_info(self, preprocessing: str, dataset_version: int) -> Tuple[Callable, list, Path]:
        if preprocessing == "raw":
            reconstruction_method = lambda data, tags, *args: data
            reconstruction_args = []
            paths = DataPaths(self.__raw_mission_path / f"V{dataset_version}" / "mission_data.arrow",
                              None,
                              self.__raw_mission_path /
                              f"V{dataset_version}" / "index_gid_map.arrow",
                              self.__raw_mission_path / f"V{dataset_version}" / "info.arrow")

        elif preprocessing != "raw" and (self.__preprocessing_path / preprocessing / "reconstruction_infos.dat").exists():
            with open(self.__preprocessing_path / preprocessing / "reconstruction_infos.dat", "rb") as reader:
                code = marshal.loads(reader.read())
            reconstruction_method = FunctionType(
                code["function"], globals(), "reconstuction_method")
            reconstruction_args = code["args"]
            paths = DataPaths(self.__raw_mission_path / f"V{dataset_version}" / "mission_data.arrow",
                              self.__preprocessing_path / preprocessing / "info.arrow",
                              self.__raw_mission_path /
                              f"V{dataset_version}" / "index_gid_map.arrow",
                              self.__raw_mission_path / f"V{dataset_version}" / "info.arrow")

        else:
            reconstruction_method = lambda data, tags, *args: data
            reconstruction_args = []
            paths = DataPaths(self.__preprocessing_path / preprocessing / "mission_data.arrow",
                              self.__preprocessing_path / preprocessing / "info.arrow",
                              self.__preprocessing_path / preprocessing / "index_gid_map.arrow",
                              self.__raw_mission_path / f"V{dataset_version}" / "info.arrow")

        return reconstruction_method, reconstruction_args, paths

    def __read_mission_info(self, dataset_version: int = None, return_version=False) -> pl.DataFrame:
        '''
        read mission base info from file accordint to dataset version. If dataset_version is None, return newes.

        Dataset contains index, gid, label, relevant and split
        '''
        # load labels from dataset_info folder by version
        dataset_versions = sorted(
            [i for i in Path(self.__raw_mission_path).iterdir() if i.is_dir()])
        assert len(
            dataset_versions) > 0, "no dataset downloaded yet! Use download-method to load dataset"

        if dataset_version is None:
            dataset_info_path = dataset_versions[-1]
        else:
            dataset_info_path = self.__raw_mission_path / f"V{dataset_version}"

        assert dataset_info_path.exists(
        ), f"Could not find dataset-version {dataset_version}"

        info = pl.read_ipc(dataset_info_path / "info.arrow")
        gid_index_map = pl.read_ipc(dataset_info_path / "index_gid_map.arrow")
        dataset_info = info.with_columns(
            gid=gid_index_map["gid"],
            index=gid_index_map["index"]
        ).select(
            pl.col("index"),
            pl.col("gid"),
            pl.col("label"),
            pl.col("relevant"),
            pl.col("split"),
        )

        if return_version:
            return dataset_info, int(str(dataset_info_path)[-1])
        else:
            return dataset_info

    def __read_mission_info_according_to_preprocessing(self, preprocessing: str = "raw", dataset_version: int = None, return_dataset_version: bool = False) -> Tuple[pl.DataFrame, DataPaths, Callable, list]:
        '''
        read the dataset-version info file and all nessasary preprocessing infos

        Dataset contains index, gid, label, relevant and split
        '''

        # read base infos
        base_infos, dataset_version = self.__read_mission_info(
            dataset_version, return_version=True)

        # load missions from correct file
        reconstruction_method, reconstruction_args, paths = self.__read_reconstruction_info(
            preprocessing, dataset_version)

        # read base infos
        base_infos = self.__read_mission_info(dataset_version)

        if preprocessing != "raw":
            # join preprocessing infos
            preprocessing_infos = pl.read_ipc(
                str(self.__preprocessing_path / preprocessing / "info.arrow"))

            joined_infos = preprocessing_infos.with_row_index().with_columns(pl.col("base_id").cast(
                pl.UInt32)).join(base_infos, left_on=pl.col("base_id"), right_on=pl.col("index"), how="left")

            if return_dataset_version:
                return joined_infos, paths, reconstruction_method, reconstruction_args, dataset_version
            else:
                return joined_infos, paths, reconstruction_method, reconstruction_args

        if return_dataset_version:
            return base_infos, paths, reconstruction_method, reconstruction_args, dataset_version
        else:
            return base_infos, paths, reconstruction_method, reconstruction_args

    def get_data(self, data: str = "split", preprocessing: str = "raw", labels: List[str] = ["test_mission", "real_mission"], gids: Optional[List[str]] = None, dataset_version: int = None, retrievel_class: DataLoader = None, lightweight: bool = False) -> DataWrapper:
        '''
        return DataLoader-Object with the selected data\\
        data can be: 
        * "all"
        * "split" default
        * "relevant"
        * None (for no sortation)

        preprocessing: kind of preprocessing

        version: 
        * None -> latest
        * int -> version 

        retrievel_class:DataLoader - Default: MemoryMappedLoader
        '''

        mission_data, paths, reconstruction_method, reconstruction_args = self.__read_mission_info_according_to_preprocessing(
            preprocessing, dataset_version)

        if retrievel_class is None:
            retrievel_class = MemoryMappedLoader(paths)
        else:
            assert retrievel_class
            retrievel_class = retrievel_class(paths)

        # determine filter based on input attributes to filter all missions by data-attributes
        if data == "all":
            query_filter = True
            groupby = "label"

        elif data == "split":
            query_filter = pl.col("relevant")
            groupby = "split"

        elif data == "relevant":
            query_filter = pl.col("relevant")
            groupby = "label"

        elif data == "None" or data == None:
            query_filter = True
            groupby = None

        else:
            raise Exception(
                f"data can be 'train', 'validation' or 'test' and not '{data}'!")

        # add gid filter
        if gids is not None:
            query_filter = query_filter & pl.col("gid").is_in(gids)

        # add label filter
        filtered_mission_data = mission_data.filter(
            query_filter & pl.col("label").is_in(labels))

        if groupby is not None:
            grouped_mission_data = dict(tuple(filtered_mission_data.select(
                pl.col("index"), groupby).group_by(groupby)))
        else:
            grouped_mission_data = {
                "data": filtered_mission_data.select(pl.col("index"))}

        grouped_mission_data = {key: np.squeeze(v.select(
            pl.col("index")).to_numpy(), 1) for key, v in grouped_mission_data.items()}

        return DataWrapper(grouped_mission_data, retrievel_class, reconstruction_method, reconstruction_args, lightweight)

    def get_mission(self, mission_gid, preprocessing: str = "raw", retrievel_class: DataLoader = None, dataset_version: int = None) -> Mission:
        '''
        Get all relevant informations for one mission.
        '''
        # load missions from correct file
        mission_data, paths, reconstruction_method, reconstruction_args = self.__read_mission_info_according_to_preprocessing(
            preprocessing, dataset_version)

        selected_mission = mission_data.filter(pl.col("gid") == mission_gid)

        assert selected_mission.shape[
            0] == 1, f"could not find any mission with gid {mission_gid}!"

        if retrievel_class is None:
            retrievel_class = MemoryMappedLoader(paths)

        return Mission(selected_mission["index"][0], retrievel_class, reconstruction_method, reconstruction_args)

    def get_uuids(self, preprocessing: str = "raw", dataset_version: int = None) -> np.ndarray:
        '''
        Get uuids for all stored missions. 
        '''
        mission_data, _, _, _ = self.__read_mission_info_according_to_preprocessing(
            preprocessing, dataset_version)

        return mission_data.to_series(1).to_numpy().astype('<U36')

    #################################################################################################################
    #                                                                                                               #
    #                                              Add missions                                                     #
    #                                                                                                               #
    #################################################################################################################

    def __store_downloaded_missions_info(self, missions: List[Tuple[int, GID, LABEL, np.ndarray]]):
        '''
        store the downloaded mission into the files
        '''
        mission_df = pl.DataFrame(
            missions, schema=["index", "gid", "label", "data"])
        mission_df = mission_df.select(
            pl.col("gid"),
            pl.col("label"),
            pl.lit(None).alias('relevant'),
            pl.lit(None).alias('split'),
            pl.col("data").apply(lambda x: x.tolist())
        )

        dataset_version = 1
        # have to read old arrow data, since file is immutable once written
        try:
            total_mission_storage, old_dataset_version = self.__read_mission_info(
                return_version=True)
            dataset_version = old_dataset_version + 1

            total_data = pl.read_ipc(
                self.__raw_mission_path / f"V{old_dataset_version}" / "mission_data.arrow")
            total_mission_storage = total_mission_storage.with_columns(
                data=total_data["data"]
            ).select(
                pl.col("gid"),
                pl.col("label"),
                pl.col("relevant"),
                pl.col("split"),
                pl.col("data")
            )

            mission_df = pl.concat([total_mission_storage, mission_df.filter(~pl.col("gid").is_in(
                total_mission_storage["gid"]))])  # concat both DF while containing order of old DF and adding new DF below
        except AssertionError:
            pass

        data_path = self.__raw_mission_path / f"V{dataset_version}"
        makedirs(data_path)

        mission_df = mission_df.with_row_index()

        # info.arrow                    contains label, relevant and split
        # index_gid_map.arrow           to map gid to index
        # mission_data.arrow            contains the recorded mission data for each index

        mission_df.select(
            pl.col("label"),
            pl.col("relevant"),
            pl.col("split")
        ).write_ipc(data_path / "info.arrow")

        mission_df.select(
            pl.col("gid"),
            pl.col("index")
        ).write_ipc(data_path / "index_gid_map.arrow")

        mission_df.select(pl.col("data")).to_pandas().to_feather(data_path /
                                                                 "mission_data.arrow")  # can not store pl.Series info arrow format :(

    def add_missions_to_database(self, missions: Union[List[ImportData], List[Tuple[GID, str, np.ndarray]]]):
        '''
        Add your non corpuls missions hier:

        Format: List[TUPLE[GID:int, LABEL:str, DATA:np.ndarray[np.float]]]
        '''
        if len(missions) < 1:
            raise Exception("missions must contain at least 1 element!")

        if type(missions[0]) is list or type(missions[0]) is tuple:
            self.__store_downloaded_missions_info(
                [(i, *d) for i, d in enumerate(missions)])

        elif type(missions[0]) is ImportData:
            self.__store_downloaded_missions_info(
                [(i, d.uid_4, d.label, d.data) for i, d in enumerate(missions)])

    #################################################################################################################
    #                                                                                                               #
    #                                    Manage everything with mission relevance                                   #
    #                                                                                                               #
    #################################################################################################################

    def __update_raw_info_file(self, dataframe: pl.DataFrame, old_dataset_version: int):
        # copy everything form old dataset to new except info.arrow
        makedirs(self.__raw_mission_path / f"V{old_dataset_version + 1}")
        copy(self.__raw_mission_path / f"V{old_dataset_version}" / "index_gid_map.arrow",
             self.__raw_mission_path / f"V{old_dataset_version +1}" / "index_gid_map.arrow")
        copy(self.__raw_mission_path / f"V{old_dataset_version}" / "mission_data.arrow",
             self.__raw_mission_path / f"V{old_dataset_version +1}" / "mission_data.arrow")

        dataframe.select(
            pl.col("label"),
            pl.col("relevant"),
            pl.col("split")
        ).write_ipc(self.__raw_mission_path / f"V{old_dataset_version +1}" / "info.arrow")

    def select_relevant_missions(self, select_method: Callable = relevant_mission, args: list = [0.5]):
        mission_data_df, paths, reconstruction_method, reconstruction_args, old_dataset_version = self.__read_mission_info_according_to_preprocessing(
            "raw", return_dataset_version=True)
        data = DataWrapper({"data": mission_data_df}, MemoryMappedLoader(
            paths), reconstruction_method, reconstruction_args, True).data

        relevance = []
        for d in tqdm(data):
            if d[0].shape[0] > 0:
                relevance.append(select_method(d[0], *args))
            else:
                relevance.append(False)

        mission_data_df = mission_data_df.with_columns(
            pl.Series(name="relevant", values=relevance)
        )

        self.__update_raw_info_file(mission_data_df, old_dataset_version)

    def get_not_relevant_missions(self, preprocessing="raw", dataset_version: int = None) -> DataWrapper:  # TODO
        # load missions from correct file

        raise Exception("not Implemented yet")

    #################################################################################################################
    #                                                                                                               #
    #                               Split Dataset into train, validation and test                                   #
    #                                                                                                               #
    #################################################################################################################

    def split_data_into_sets(self, split_method: Callable[[List[Any], [...]], Dict[str, Tuple[str, str]]] = split_function, args: list = [[0.6, 0.1, 0.3]]):
        mission_data_df, paths, reconstruction_method, reconstruction_args, old_dataset_version = self.__read_mission_info_according_to_preprocessing(
            "raw", return_dataset_version=True)

        real = split_method(mission_data_df.filter(pl.col("label") == "real_mission")[
                            "index"].to_numpy().copy(), *args)
        test = split_method(mission_data_df.filter(pl.col("label") == "test_mission")[
                            "index"].to_numpy().copy(), *args)

        new_split = [None]*mission_data_df.shape[0]
        for i in [real, test]:
            for k, v in i.items():
                for index in v:
                    new_split[index] = k

        mission_data_df = mission_data_df.with_columns(
            pl.Series(name="split", values=new_split)
        )

        self.__update_raw_info_file(mission_data_df, old_dataset_version)

    #################################################################################################################
    #                                                                                                               #
    #                                     Preprocess data and correct labels                                        #
    #                                                                                                               #
    #################################################################################################################

    def correct_mission_label(self, new_labels: Dict[str, str]):
        '''
        This method corrects the mission labels for raw, relevant and split. \n
        Input:\n
            new_labels: Dict[mission_gid, label]\n
                label can be "real_missions" or "test_missions"
        '''

        # switch labels in all missions
        mission_data_df, paths, reconstruction_method, reconstruction_args, old_dataset_version = self.__read_mission_info_according_to_preprocessing(
            "raw", return_dataset_version=True)

        mission_data_df = mission_data_df.map_rows(lambda x: (x[0], x[1], x[2] if x[1] not in new_labels else new_labels[x[1]], x[3], x[4])).select(
            pl.col("column_0").alias("index"),
            pl.col("column_1").alias("gid"),
            pl.col("column_2").alias("label"),
            pl.col("column_3").alias("relevant"),
            pl.col("column_4").alias("split")
        )  # TODO can be improfed :)

        self.__update_raw_info_file(mission_data_df, old_dataset_version)

    def preprocess_data(self, preprocessing_name: str, data: Union[List[Tuple[AbstractDataClass, int]], DataWrapper], preprocess_method: Callable, args: list, recreation_method: Callable = None, recreation_args: list = None):
        '''
        if recreation_method is set, the intermediate numpy results will not be stored, insted on retriving the preprocessed data, this method will be used to load the preprocessed data using the aditional informations stored in the tags.
        '''

        if type(data) is DataWrapper:
            mission_data = list(chain(*[[(k[:-8], i) for i in getattr(data, k)]
                                for k in data.__dict__.keys() if k[-8:] == "_indices"]))
        elif type(data) is list:
            mission_data = data
        else:
            raise AssertionError(
                "Not supported type of data attribut! Supportet ist List of Iterable and DataLoader-Object")

        preprocessing_path = self.__preprocessing_path / preprocessing_name

        assert not preprocessing_path.exists(), "this preprocessing allready exists!"

        makedirs(preprocessing_path)

        if recreation_method is not None:
            with open(preprocessing_path / "reconstruction_infos.dat", "wb") as writer:
                marshal.dump({"function": recreation_method.__code__,
                             "args": recreation_args}, writer)

        data_df = pl.DataFrame(None, schema=["base_id", "id", "args"]).select(pl.col("base_id").cast(
            pl.Int64), pl.col("id").cast(pl.String), pl.col("args").cast(pl.List(pl.Int64)))
        if recreation_method is not None:
            for attribute, index in tqdm(mission_data):
                m_data = getattr(data, attribute).get_by_index(index)[0]
                preprocess_data = preprocess_method(m_data, *args)

                if type(preprocess_data) is list:
                    data_df = data_df.vstack(pl.DataFrame([[index, str(uuid.uuid4(
                    )), tags] for _, tags in preprocess_data], schema=["base_id", "id", "args"]))
                else:
                    data_df = data_df.vstack(pl.DataFrame(
                        [[index, str(uuid.uuid4()), preprocess_data[1]]], schema=["base_id", "id", "args"]))
        else:
            with pa.OSFile(preprocessing_path / "mission_data.arrow", 'wb') as sink:
                with pa.ipc.new_file(sink, schema=PA_SCHEMA) as writer:
                    for attribute, index in tqdm(mission_data):
                        m_data = getattr(
                            data, attribute).get_by_index(index)[0]
                        preprocess_data = preprocess_method(m_data, *args)

                        if type(preprocess_data) is list:
                            for p_data, tags in preprocess_data:
                                batch = pa.record_batch(
                                    [pa.array(p_data)], schema=PA_SCHEMA)
                                writer.write(batch)
                                data_df = data_df.vstack(pl.DataFrame([[index, str(uuid.uuid4(
                                )), tags] for _, tags in preprocess_data], schema=["base_id", "id", "args"]))
                        else:
                            batch = pa.record_batch(
                                [pa.array(preprocess_data[0])], schema=PA_SCHEMA)
                            writer.write(batch)
                        data_df = data_df.vstack(pl.DataFrame(
                            [[index, str(uuid.uuid4()), preprocess_data[1]]], schema=["base_id", "id", "args"]))

        data_df = data_df.with_row_index()

        data_df.select(pl.col("base_id"), pl.col("args")).write_ipc(
            preprocessing_path / "info.arrow")
        data_df.select(pl.col("id"), pl.col("index")).write_ipc(
            preprocessing_path / "index_gid_map.arrow")

    def list_preprocessings(self) -> List[str]:
        return [i.name for i in self.__preprocessing_path.iterdir() if i.is_dir()]

    #################################################################################################################
    #                                                                                                               #
    #                                        Manage multiple databases                                              #
    #                                                                                                               #
    #################################################################################################################

    def switch_database(self, name: Optional[str] = None, path: Optional[Union[Path, str]] = None):
        '''
        switch database to ether path or name of existing database. If none is declared, switch to base
        '''
        if path is not None:
            path = Path(path) if type(path) is not Path else path
            assert path.exists(), f"No such Database '{str(path)}'!"
        elif name is not None:
            with open(Path(__file__).parent / "data_path.yaml", "r") as file:
                if name == "base":
                    path = Path(yaml.safe_load(file)["DATA_PATH"])
                else:
                    d_bases = yaml.safe_load(file)["DATABASES"]
                    if name in d_bases.keys():
                        path = Path(d_bases[name])
                    else:
                        raise AssertionError(
                            f"Could not find a database with name '{name}'!")
        else:
            print("Switching database to 'base'.")
            with open(Path(__file__).parent / "data_path.yaml", "r") as file:
                path = Path(yaml.safe_load(file)["DATA_PATH"])

        self.__local_path = path
        self.__raw_mission_path = self.__local_path / "raw"
        self.__preprocessing_path = self.__local_path / "preprocessing"

    def setup_new_database(self, name: str, path: Union[Path, str], mission_gids: List[GID]):
        path = Path(path) if type(path) is not Path else path
        assert not path.exists(), f"Database '{str(path)}' already exits"

        # create new base_path
        makedirs(path)

        # get missions
        missions = self.get_data(data="all", gids=mission_gids)

        # switch paths
        self.switch_database(path)

        # create new folders
        makedirs(self.__raw_mission_path)
        makedirs(self.__preprocessing_path)

        # add missions to new database
        missions_to_add = []
        for m_data in vars(missions).keys():
            for data in getattr(missions, m_data):
                missions_to_add.append((data.gid, data.label, data.data))

        if len(missions_to_add) > 0:
            self.add_missions_to_database(missions_to_add)
            self.select_relevant_missions(lambda x: True, [])

        # add new database to
        data_path = Path(__file__).parent / "data_path.yaml"

        with open(data_path, 'r') as file:
            data = yaml.safe_load(file)
            # base data path
            base_path = data["DATA_PATH"]
            # load other existing databases
            if "DATABASES" not in data.keys():
                existing_databases = {}
            else:
                existing_databases = data["DATABASES"]

        # add path of new database
        existing_databases[name] = str(path)

        # store new database path
        with open(data_path, 'w') as file:
            yaml.safe_dump(
                {"DATA_PATH": base_path, "DATABASES": existing_databases}, file)

    def list_databases(self):
        '''
        list all available databases.
        '''
        data_path = Path(__file__).parent / "data_path.yaml"

        with open(data_path, 'r') as file:
            data = yaml.safe_load(file)
            if "DATABASES" not in data.keys():
                existing_databases = {}
            else:
                existing_databases = data["DATABASES"]

        return ["base"] + list(existing_databases.keys())

    def remove_database(self, path: Optional[Union[Path, str]] = None, name: Optional[str] = None):
        '''
        switch database to ether path or name of existing database
        '''
        if path is not None:
            path = Path(path) if type(path) is not Path else path
            assert (path / "raw").exists(), f"No such Database '{str(path)}'!"
            with open(Path(__file__).parent / "data_path.yaml", "r") as file:
                assert yaml.safe_load(file)["DATA_PATH"] != str(
                    path), "You can not remove base database!"
        elif name is not None:
            with open(Path(__file__).parent / "data_path.yaml", "r") as file:
                if name == "base":
                    raise AssertionError("You can not remove base database!")
                else:
                    d_info = yaml.safe_load(file)
                    if name in d_info["DATABASES"].keys():
                        path = Path(d_info["DATABASES"][name])

                        # remove path from data_paths
                        del d_info["DATABASES"][name]
                        with open(Path(__file__).parent / "data_path.yaml", 'w') as file:
                            yaml.safe_dump(
                                {"DATA_PATH": d_info["DATA_PATH"], "DATABASES": d_info["DATABASES"]}, file)

                    else:
                        raise AssertionError(
                            f"Could not find a database with name '{name}'!")

        if str(path) == str(self.__local_path):
            self.switch_database(name="base")

        try:
            rm_dirs(str(path))
        except Exception as e:
            print(f"Could not delete '{path}' completly because of Error: {e}")

    def add_existing_databases(self, path: Union[Path, str]):
        path = Path(path) if type(path) is not Path else path
        assert path.exists(), f"Path '{str(path)}' does not exist!"

        # get path of data info
        data_path = Path(__file__).parent / "data_path.yaml"

        dataset_paths = next(os.walk(path))[1]

        new_data_paths = {}
        for p in dataset_paths:
            if (path / p / "database_info.yaml").exists() and (path / p / "preprocessing/").exists() and (path / p / "raw/").exists():
                with open(path / p / "database_info.yaml", "r") as reader:
                    new_data_paths[yaml.safe_load(
                        reader)["NAME"]] = str(path / p)

        with open(data_path, 'r') as file:
            data = yaml.safe_load(file)
            # base data path
            base_path = data["DATA_PATH"]
            # load other existing databases
            if "DATABASES" not in data.keys():
                existing_databases = {}
            else:
                existing_databases = data["DATABASES"]

        with open(data_path, 'w') as file:
            yaml.safe_dump(
                {"DATA_PATH": base_path, "DATABASES": existing_databases | new_data_paths}, file)

    #################################################################################################################
    #                                                                                                               #
    #                                           Dataset statistics                                                  #
    #                                                                                                               #
    #################################################################################################################

    def statistics(self, preprocessing="raw", dataset_version: int = None) -> Dict[str, Dict[str, int]]:
        # load missions from correct file

        info, _, _, _ = self.__read_mission_info_according_to_preprocessing(
            preprocessing, dataset_version)

        relevant_missions = info.filter(pl.col("relevant") == True)

        train_split_missions = relevant_missions.filter(
            pl.col("split") == "train")
        validate_split_missions = relevant_missions.filter(
            pl.col("split") == "validation")
        test_split_missions = relevant_missions.filter(
            pl.col("split") == "test")

        basic = {"all": {"real_missions": info.filter(pl.col("label") == "real_mission").select(pl.count()).item(
        ), "test_missions": info.filter(pl.col("label") == "test_mission").select(pl.count()).item()}}

        basic["relevant"] = {"real_missions": relevant_missions.filter(pl.col("label") == "real_mission").select(pl.count(
        )).item(), "test_missions": relevant_missions.filter(pl.col("label") == "test_mission").select(pl.count()).item()}

        basic["split"] = {"train": {"real_missions": train_split_missions.filter(pl.col("label") == "real_mission").select(
            pl.count()).item(), "test_missions": train_split_missions.filter(pl.col("label") == "test_mission").select(pl.count()).item()}}

        basic["split"]["validation"] = {"real_missions": validate_split_missions.filter(pl.col("label") == "real_mission").select(
            pl.count()).item(), "test_missions": validate_split_missions.filter(pl.col("label") == "test_mission").select(pl.count()).item()}

        basic["split"]["test"] = {"real_missions": test_split_missions.filter(pl.col("label") == "real_mission").select(
            pl.count()).item(), "test_missions": test_split_missions.filter(pl.col("label") == "test_mission").select(pl.count()).item()}

        return basic
