# data_out directory

Directory for storing data output by the program.

When `FileController.export_data()` is used to export data, this directory is automatically used as the destination.
The directory is divided to organize the data and does not need to be used.

## Output data name

Basically, the name of the output data includes the name of the file to be analyzed.
In addition, to prevent accidental deletion of previous data during program execution, the output data is prefixed with the date and time of execution. 
The analyzed contents and parameters are suffixed.