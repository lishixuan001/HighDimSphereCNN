# PointCloud Classification & Segmentation With Manifold Spherical CNNs
> Mapping object into high dimensional spheres, with relative position attributes intact, to perform manifold CNNs

## 1 Project layout
```
Project
│
└───ICSI-Vision-HighDimSphereCNN/
│   │   data_generation.py
│   │   helper_data_generation.py
│   │   train.py
│   │   helper_train.py
│   │   model.py
│   │   wFM.py
│   │   utils.py
│   │   logger.py
│   │   
│   └───log/
│       │   2019-02-23-08:29:12.log
│       │   2019-02-23-09:01:24.log
│       │   ...
│   
└───mnistPC/
    │   test.hdf5
    │   train.hdf5
    │   test.gz
    │   train.gz
    │   demo_test.gz
    │   demo_train.gz
    │   ...
```

### 1.1 Source Dataset
Source data are stored in `../mnistPC/` directory

### 1.2 Log
Log infos are stored in `./log/` directory

## 2 Source Code
### 2.1 Data Generation
> Data generation mainly in charge loading data (point_cloud, labels) from the `.hdf5` files, and map them into manifold space, then store the generated data via `pickle` and save as `.gz` files in `../mnistPC/` directory

* Involved scripts: **data_generation.py**, **helper_data_generation.py**
* Go by the name, **data_generation.py** mainly present the structure of the data generation, while **helper_data_generation.py** provides the functional methods for each procedure
#### 2.1.1 *data_generation.py*
1. Parse user's argument input (batch_size, test, demo, etc)
2. Load data from `.hdf5` files (data, labels)
3. Compute adjacent matrix 
4. Perform data normalization, project the object point cloud data into `[-1, 1]` 3D space
5. Create grid according to the space (`grid_size` determined by user specification)
6. Map the projected data space into a high dimensional spherical manifold surface
7. Use DatasetConstructor to store these data, serialize it, and save it as `.gz` file in corresponding dataset source directory

#### 2.1.2 *helper_data_generation.py*
* [Class] `DatasetConstructor`
* [Method] `load_args()`
* [Method] `raw_data_normalization(-)`
* [Method] `grid_generation(-)`
* [Method] `map_and_norm(-)`
* [Helper] `progress(-)`

### 2.2 Training
> Training process mainly pass data into the CNNs, consistently train the model and keep track of the loss and accuracy. 

* Involved scripts: **train.py**, **helper_train.py**
* Go by name, **train.py** mainly defines the procedures of training, while **helper_train.py** provides functional methods for computations

#### 2.2.1 *train.py*
1. Load user arguments inputs
2. Load data from the `.gz` files, get the `DataConstructor` type serialized class containing dataset information from data generation processes, including *normalization->mapping->normalization* generated data, labels, and adjacent matrix
3. Eventually load data into `torch.utils.data.DataLoader` constructor
4. Iterate through `epochs` and by `batchs`, keep track of the loss/accuracy value, and consistently report log info

#### 2.2.2 *helper_train.py*
1. [Method] `load_args()`
2. [Method] `load_data(-)`
3. [Method] `evaluate(-)`
4. [Helper] `progress`

### 2.3 Model
**[Under Construction]**

