# LocalDataManagementTool

Package to manage local datasets. Developed for corpuls to process corpulsAED-missions.

## Build package

This package is not yet available via a pip repository server. Therefor you have to clone this repository and build it yourself.

To build the repository you can use the pip package 'build'
```console
pip install build
```
and build the package with `python -m build`. The results of the build process are within the folder /dist. You can install this local package with pip:
```console
pip install /path/to/repository/dist/localdatasetmanagement-X.X.X-py3-none-any.whl
```


## Initialize

```python
from localdataset.dataset import LocalDataset

ds = LocalDataset("/path/where/data/is/stored")
```
The data path will be stored an when not specified otherwise be used.

### Download missions

Add missions with the method 'add_missions'

```python
def add_missions_to_database(self, missions: List[Tuple[<GID:str>, <label:str>, np.ndarray]])
``` 

### Acess data

You can access the data via the method:

```python
def get_data(
  data: str = "split",
  preprocessing: str = "raw",
  labels: List[str] = ["test_mission", "real_mission"],
  gids: Optional[List[str]] = None,
  dataset_version: int = None,
  retrievel_class: DataLoader = None,
  lightweight: bool = False,
  ) -> DataWrapper:
```

Data can be accessed via the DataWrapper object. Depending on which data -- split, relevant or all -- you want to select, the object provides corresponding attributes. These attributes contain ether a Data or lightWeightData object. The missions can be accessed via the get_by_index using the indices availabe through the indices attribute.

To get the data of one mission by gid use 

```python
def get_mission(
  mission_gid,
  preprocessing: str = "raw",
  retrievel_class: DataLoader = None,
  dataset_version: int = None,
  ) -> Mission:
```