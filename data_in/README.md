# data_in directory

Directory for storing data to be entered into the program．

Passing this directory to `FileController.get_file()` as an argument makes it easier to select files in the directory via GUI.

## CSV Data Details

- 1111_l123r123_q1_md20_ws15_label: 6 classifications, labels
- 1111_l123r123_q1_md20_ws15_fixed: 6 classification, raw data
- 1111_l123r123_q1_md20_ws15_fixed_std: 6 classification, standardized data
- 1122_a000-315xw147_q1_md20_ws15_fixed: 24 classifications, raw data
- 1122_a000-315xw147_q1_md20_ws15_label: 24 classifications, labels

(Since the labels are a 0,1 matrix, they may be generated programmatically.)

## Meaning of input data names

- lr1lr2lr3, l123r123 (6 class): The order in which the videos are concatenated is shown, where l and r represent the wind direction and the numbers represent the wind speed level.
As an example, l123r123 is the feature extracted from the videos concatenated in the order shown in the table below.

|concatenation order|wind direction|Wind Speed Level|
|:---:|:---:|---:|
|1|left|1|
|2|left|2|
|3|left|3|
|4|right|1|
|5|right|2|
|6|right|3|

- a000-315xw147 (24 class): Wind direction was varied from 0° to 315° by 45°.
wind speed was varied at 1, 4, and 7 (wind speed levels 1, 2, and 3).


|concatenation order|wind direction(degree)|Wind Speed Level|
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

- q1: Parameters for feature point extraction (threshold for accepting detected points as feature points q=0.1)
- md20: Parameters for feature point extraction (distance between feature points 20px)
- ws15: Parameters for feature point extraction (Window size at detection: 15px)
- suffix (The part beginning with the trailing `_`.): The following table shows.

|suffix|content|
|---|---|
|`_fixed`|Data extracted only from the points on the plant from all detected feature points (modified data)|
|`_label`|Label corresponding to the data|
|`_std`|standardized data|
|`_norm`|normalized data|