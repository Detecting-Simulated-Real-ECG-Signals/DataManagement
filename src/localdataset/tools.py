from typing import Any, Dict, List, Tuple, Union

import numpy as np

from .dataset import relevant_mission

TAGS = Dict[str, Any]
GID = str


class RelevantWindowV1:
    '''
    select relevant windows by spectral analysis using the fourier transformation.
    '''

    @staticmethod
    def relevant_determination(window: np.ndarray, relevant_windows_threshold: float) -> bool:
        return relevant_mission(window, relevant_windows_threshold)

    @staticmethod
    def preprocess_mission_window(data: np.ndarray, window_size: int, hist_threshold: float = 0.3, relevant_windows_threshold: float = 0.5) -> Union[Tuple[np.ndarray, List[int]], List[Tuple[np.ndarray, Dict[str, Any]]]]:
        if data.shape[0] >= window_size:
            windows = np.lib.stride_tricks.sliding_window_view(
                data, window_size)
            # invertion of relevant missions becouse in the next code row the rows with true will be deleted
            delete_rows = np.apply_along_axis(lambda x: not RelevantWindowV1.relevant_determination(
                x, relevant_windows_threshold), axis=1, arr=windows)

            preprocessed_data = np.delete(windows, delete_rows, 0)

            return [(preprocessed_data, [ts]) for preprocessed_data, ts in list(zip(preprocessed_data, *np.invert(delete_rows).nonzero()))]
        else:
            return []

    @staticmethod
    def regenerate_mission_windows_from_tag(data: np.ndarray, tags: List[int], window_size: int) -> np.ndarray:
        return data[tags[0]:tags[0]+window_size]


class RelevantWindowV2:
    '''
    select relevant windows by determining var for the first 10 and last 10 measuerement points. If both vars are not 0, window is relevant
    '''
    @staticmethod
    def relevant_determination(window: np.ndarray, bounds: list, var_threshold) -> bool:
        return window[:bounds[0]].var() > var_threshold and window[bounds[1]:].var() > var_threshold

    @staticmethod
    def preprocess_mission_window(data: np.ndarray, window_size: int, var_threshold: float = 0.2, bounds=[10, 10]) -> Union[Tuple[np.ndarray, List[int]], List[Tuple[np.ndarray, Dict[str, Any]]]]:

        if data.shape[0] >= window_size:
            windows = np.lib.stride_tricks.sliding_window_view(
                data, window_size)
            # invertion of relevant missions becouse in the next code row the rows with true will be deleted
            delete_rows = np.apply_along_axis(lambda x: not RelevantWindowV2.relevant_determination(
                x, bounds, var_threshold), axis=1, arr=windows)

            preprocessed_data = np.delete(windows, delete_rows, 0)

            return [(preprocessed_data, [ts]) for preprocessed_data, ts in list(zip(preprocessed_data, *np.invert(delete_rows).nonzero()))]
        else:
            return []

    @staticmethod
    def regenerate_mission_windows_from_tag(data: np.ndarray, tags: List[int], window_size: int) -> np.ndarray:
        return data[tags[0]:tags[0]+window_size]
