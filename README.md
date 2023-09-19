## environment construction

The program has been tested with Python 3.7.2.

If you can use Poetry, you can install the package as follows.

```zsh
$ git clone https://github.com/mktia/plant_reservoir.git && cd ./plant_reservoir
$ poetry install
```

Even if you are not in the Poetry environment, you can check for and install the necessary packages using pyproject.toml > `[tool.poetry.dependencies]`.

## Summary of each file

- /data_in: input data
- /data_out: output data
- /image_out: output image
- /video_in: input Video
- /video_out: output Video
- [classify.py](classify.py): verification of classifications and their results (Main)
- [csv_manager.py](csv_manager.py): CSV preprocessing
- [detect_corners_by_gftt.py](detect_corners_by_gftt.py): Confirmation of feature points detected by Shi-Tomasi Corner Detector (still image)
- [detect_features.py](detect_features.py): Verification of other feature point detection methods that can be realized with OpenCV (video)
- [gp.py](gp.py): Calculation of the correlation dimension using the Grassberger-Procaccia method
- [gp_cupy.py](gp_cupy.py): Calculation of correlation dimension by GP method (using GPU)
- [nn.py](nn.py): Class classification with simplified ResNet
- [nn_gpu.py](nn_gpu.py): Class classification with simplified ResNet (using GPU)
- [optical_flow_gftt.py](optical_flow_gftt.py): Object Tracking by Optical Flow
- [pca_dim.py](pca_dim.py): Verification of relationship between distance between distributions and cross-entropy by PCA, EMD, and Ward methods
- [plot_on_video.py](plot_on_video.py): Wout reflected in the video
- [pooling.py](pooling.py): Video size reduction by max pooling
- [utils.py](utils.py): Processing assistance
- [utils_cp.py](utils_cp.py): Processing assistance (using GPU)
- [video_manager.py](video_manager.py): Splitting and concatenating videos, etc.

## Use of GPU

The files marked as (GPU used) are prepared for high-speed execution on a GPU environment running CUDA, and require the cupy package to be installed in the execution environment.

Note: Sometimes utils_cp.py is required for execution (gp_cupy.py).

## How to use the code

### Figure 2 (a), (b)

source: `regression_6class()` in [classify.py](classify.py)

```
$ python classify.py
```

In the dialog that appears in the terminal after execution, enter
0 (Classification with 6 classes).

### Figure 3

source `relation_between_unit_size_and_accuracy()` in [classify.py](classify.py)

```
$ python classify.py
```

In the dialog that appears in the terminal after execution, enter 2 (Relation between Number of units and accuracy).

### Figure 4 (a), (b)

source: `regression_24class()` in [classify.py](classify.py)

```
$ python classify.py
```

In the dialog that appears in the terminal after execution, enter 1 (Classification with 24 classes).

### Figure 5 (a)-(f)

source: `untrained()` in [classify.py](classify.py)

```
$ python classify.py
```

In the dialog that appears in the terminal after execution, enter Enter 10 (untrained).
Also, specify the classes to be unlearned by entering a number.

### Figure 6

source: `robustness()` in [classify.py](classify.py)

```
$ python classify.py
```

In the dialog that appears in the terminal after execution, enter 11 (robustness).


### Figure 7

source: `delay_expansion()` in [classify.py](classify.py)

```
$ python classify.py
```

In the dialog that appears in the terminal after execution, enter 7 (delay expansion).


### Figure 8

source: `relation_between_unit_size_and_accuracy_by_logistic()` in [classify.py](classify.py)

```
$ python classify.py
```

In the dialog that appears in the terminal after execution, enter 5 (Relation between Number of units and accuracy by logistic regression).


### Supplementary Figure 1 (a), (b)

source: `regression_6class()` in [classify.py](classify.py)

```
$ python classify.py
```

In the dialog that appears in the terminal after execution, enter 0 (Classification with 6 class).


### Supplementary Figure 2 (a), (b)

source: `regression_24class()` in [classify.py](classify.py)

```
$ python classify.py
```

In the dialog that appears in the terminal after execution, enter 1 (Classification with 24 class) .

### Supplementary Figure 3

source: [3d_graph.py](3d_graph.py)

```
$ python 3d_graph.py
```

## Data

- Video: https://drive.google.com/drive/folders/1LWd5YxdaYgV7LBGYyNsZEzSrYIuz-2wU?usp=sharing
- CSV: https://drive.google.com/drive/folders/1YNS0ZcgT-JnNjyp5C_0DHVdh4kBL0n0-?usp=sharing