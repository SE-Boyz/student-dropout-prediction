# Dataset Instructions

This project uses the UCI dataset:
- Title: Predict Students' Dropout and Academic Success
- URL: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
- DOI: https://doi.org/10.24432/C5MC89

Required filename in this folder:
- `student_dropout_academic_success.csv`

The preprocessing notebook will automatically:
- inspect the dataset
- summarize feature types and class distribution
- split numeric and categorical variables appropriately
- encode categorical variables
- standardize numeric variables
- perform a stratified 80/20 train-test split
- save `X_train.csv`, `X_test.csv`, `y_train.csv`, and `y_test.csv`
