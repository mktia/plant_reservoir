# data_in directory

プログラムに入力するデータを格納するディレクトリ．

`FileController.get_file()` にこのディレクトリを引数で渡すと，GUIでディレクトリ内部のファイルを選択しやすくなる．

## CSVデータ詳細

- 1111_l123r123_q1_md20_ws15_label: 6クラス分類，ラベル
- 1111_l123r123_q1_md20_ws15_fixed: 6クラス分類，生データ
- 1111_l123r123_q1_md20_ws15_fixed_std: 6クラス分類，標準化済みデータ
- 1122_a000-315xw147_q1_md20_ws15_fixed: 24クラス分類，生データ
- 1122_a000-315xw147_q1_md20_ws15_label: 24クラス分類，ラベル

（ラベルは0,1の行列であるため，プログラム上で生成しても良い）

## 入力データの名称の意味

- lr1lr2lr3, l123r123 (6 class): 動画の連結順を示しており，lとrは風向，数字は風速レベルを表す．
例として，l123r123は以下の表に示す順番で連結した動画から特徴量を抽出したものである．

|連結順|風向|風速レベル|
|:---:|:---:|---:|
|1|左|1|
|2|左|2|
|3|左|3|
|4|右|1|
|5|右|2|
|6|右|3|

- a000-315xw147 (24 class): 風向を0°から315°まで45°ずつ，
風速を1, 4, 7（風速レベル1, 2, 3）で変化させたときのデータである．

|連結順|風向 (度)|風速レベル|
|:---:|---:|---:|
|1|0|1|
|2|0|2|
|3|0|3|
|4|45|1|
|5|45|2|
|6|45|3|
|7|90|1|
|8|90|2|
|9|90|3|
|10|135|1|
|11|135|2|
|12|135|3|
|13|180|1|
|14|180|2|
|15|180|3|
|16|225|1|
|17|225|2|
|18|225|3|
|19|270|1|
|20|270|2|
|21|270|3|
|22|315|1|
|23|315|2|
|24|315|3|

- q1: 特徴点抽出時のパラメータ（検出した点を特徴点と認める閾値 q=0.1）
- md20: 特徴点抽出時のパラメータ（特徴点間距離 20px）
- ws15: 特徴点抽出時のパラメータ（検出時のウィンドウサイズ 15px）
- suffix (最後尾の`_`に始まる部分): 以下の表に示す

|suffix|content|
|---|---|
|`_fixed`|検出した全特徴点から植物上の点のみ抽出したデータ（修正済みデータ）|
|`_label`|データに対応するラベル|
|`_std`|標準化済みデータ|
|`_norm`|正規化済みデータ|