'''
Module contains data types used by the LocalDataManagement tool
'''

from pathlib import Path
from typing import List, NamedTuple
from uuid import UUID

# Types
INDICES = List[int]
LABEL = str
INDEX = int
GID = UUID
TAGS = List[int]


class DataPaths(NamedTuple):
    data: Path
    info: Path
    uid: Path
    label: Path
