# LocalDataManagementTool

Package to manage local datasets. Developed for corpuls to process CAED-missions.

## Build package

This package is not published yet. Therefor you have to clone this repository and build it yourself.

To build the repository you can use the pip package 'build'
```console
pip install build
```
and build the package with `python -m build`. The results of the build process are within the folder /dist. You can install this local package with pip:
```console
pip install /path/to/localdatasetmanagement-X.X.X-py3-none-any.whl
```


## Initialize

```python
from localdataset.dataset import LocalDataset

ds = LocalDataset("/path/where/data/is/stored")
```
The data path will be stored an when not specified otherwise be used.

### Download missions

Download missions from corpuls analyse use ether 'download_mission_with_query'

```python
ds.download_mission_with_query("sql query", corpuls_dataset, store_mission_info=True)
``` 

To download missions of another analyse server use this method. For this you need to allready know the gids!

```python
ds.download_mission_by_files(missions:list[gids], analyse_server:str="https://gs-pseudomissions.corpulsweb.com/corpuls.web/analyse", credentials:Path=Path("credentials.yaml"), store_mission_info=True)
```
Credentials must contain 'USERNAME' and 'PASSWORD'.

To download both from default analyse server and from an other one you can use this method:

```python
def download(self, missions:list, query:str, dataset=dataset("public"), analyse_server:str="https://gs-pseudomissions.corpulsweb.com/corpuls.web/analyse", credentials:Path=Path("credentials.yaml")):
```

### Acess data

You can access the data via the method:

```python
ds.get_data()
```

To get the data of one mission by gid use 

```python
ds.get_mission()
```