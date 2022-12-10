# Predict Customer Churn

- Project **Predict Customer Churn** using pytest, pip8.

## Project Description
This project predicts the probability of churning using the bank dataset


## Files and data description
The entry file is `churn_library.py`. It will run all the necessary functions and will create a log under logs/results.log.

For running the tests, just need to run `pytest`. Pytest will complain if data or files are not present, also will assert basic aspects of the df such as column names.

As this project was build locally, I collected all required packages using a conda environment. See and install `requirements.txt` for replicating it in your machine. (pip list -format -freeze > requirements.txt)

## Running Files
You need to run `pytest` for testing, if you run `python test_churn_script_logging.py` you'll see the log file will be populated
`python churn_library.py`
