## 環境構築

Python 3.7.2 で実行確認済み．

Poetry が使えるのであれば以下の要領でパッケージをインストール可能．

```zsh
$ git clone https://github.com/mktia/plant_reservoir.git && cd ./plant_reservoir
$ poetry install
```

Poetry 環境でなくても pyproject.toml > `[tool.poetry.dependencies]` で必要なパッケージを確認してインストールすれば良い．

## 各ファイルの概要

- /data_in: 入力データ
- /data_out: 出力データ
- /image_out: 出力画像
- /video_in: 入力動画
- /video_out: 出力動画
- [classify.py](classify.py): クラス分類やその結果に関する検証（メイン）
- [csv_manager.py](csv_manager.py): CSVの前処理
- [detect_corners_by_gftt.py](detect_corners_by_gftt.py): Shi-Tomasi Corner Detectorで検出した特徴点の確認（静止画）
- [detect_features.py](detect_features.py): OpenCVで実現できるその他の特徴点検出手法の検証（動画）
- [gp.py](gp.py): Grassberger-Procaccia法による相関次元の算出
- [gp_cupy.py](gp_cupy.py): GP法による相関次元の算出（GPU使用）
- [nn.py](nn.py): 簡易版ResNetによるクラス分類
- [nn_gpu.py](nn_gpu.py): 簡易版ResNetによるクラス分類（GPU使用）
- [optical_flow_gftt.py](optical_flow_gftt.py): Optical Flow による物体追跡
- [pca_dim.py](pca_dim.py): PCA, EMD, Ward法による分布間距離と交差エントロピーの関係性の検証
- [plot_on_video.py](plot_on_video.py): 動画にWoutを反映
- [pooling.py](pooling.py): 最大プーリングによる動画サイズの縮小
- [utils.py](utils.py): 処理の補助
- [utils_cp.py](utils_cp.py): 処理の補助（GPU使用）
- [video_manager.py](video_manager.py): 動画の分割・連結など

## GPUの使用について

（GPU使用）と書かれているファイルは CUDA が動いている GPU 環境で
高速実行するために用意したファイルで，動作させるには実行環境で cupy パッケージのインストールが必要．

Note: 実行に utils_cp.py が必要なことがある（gp_cupy.py）．

## コードの使用方法

### Figure 2 (a), (b)

source: `regression_6class()` in [classify.py](classify.py)

```
$ python classify.py
```

実行後にターミナルに表示されるダイアログで
0 (Classification with 6 class) を入力する．

### Figure 3

source `relation_between_unit_size_and_accuracy()` in [classify.py](classify.py)

```
$ python classify.py
```

実行後にターミナルに表示されるダイアログで
2 (Relation between Number of units and accuracy) を入力する．

### Figure 4 (a), (b)

source: `regression_24class()` in [classify.py](classify.py)

```
$ python classify.py
```

実行後にターミナルに表示されるダイアログで
1 (Classification with 24 class) を入力する．

### Figure 5 (a)-(f)

source: `untrained()` in [classify.py](classify.py)

```
$ python classify.py
```

実行後にターミナルに表示されるダイアログで
10 (untrained) を入力する．
また，学習しないクラスは数字で指定する．


### Figure 6

source: `robustness()` in [classify.py](classify.py)

```
$ python classify.py
```

実行後にターミナルに表示されるダイアログで
11 (robustness) を入力する．

### Figure 7

source: `delay_expansion()` in [classify.py](classify.py)

```
$ python classify.py
```

実行後にターミナルに表示されるダイアログで
7 (delay expansion) を入力する．

### Figure 8

source: `relation_between_unit_size_and_accuracy_by_logistic()` in [classify.py](classify.py)

```
$ python classify.py
```

実行後にターミナルに表示されるダイアログで
5 (Relation between Number of units and accuracy by logistic regression) を入力する．
