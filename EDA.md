# EDA Notes

## Current status
- The project structure has been created under the workspace.
- The exploration notebook is available at notebooks/EDA.ipynb.
- The raw dataset folder contains train.csv and test.csv, but both files are currently empty (0 bytes each).

## What was corrected
- The earlier notebook instructions assumed a dataset file already existed.
- In this workspace, the raw CSV files are present as placeholders only, so pandas cannot read them yet.
- The notebook now checks the corrected paths and reports the issue clearly instead of failing with a file-not-found-style error.

## What is needed next
- Add a real TON-IoT CSV file to dataset/raw, such as:
  - TON_IoT.csv
  - Train_Test_Network.csv
  - train.csv (with real data)
  - test.csv (with real data)

## Expected next step after data is added
- Run the notebook cells to inspect:
  - shape
  - head
  - columns
  - info
  - missing values
  - target column distribution
  - numerical vs categorical features
