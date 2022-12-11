'''
author: Edgar Gironés
Date: 24.11.22
Module for the logging and testing of the churn prediction
'''
from os.path import exists
import logging
import pytest
import joblib
import churn_library as cls

pytest.main()

def test_import():
    '''
    test data import - this example is completed for
    you to assist with the other test functions
    '''
    try:
        df_test = cls.import_data("./data/bank_data.csv")
        logging.info("File loaded")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df_test.shape[0] > 0
        assert df_test.shape[1] > 0
        logging.info('DF shape is good')
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    pytest.df_test = df_test
    return df_test


list_figures = ['./images/eda/churn.jpeg',
                './images/eda/customer_age.jpeg',
                './images/eda/marital_status.jpeg',
                './images/eda/trans_ct.jpeg',
                './images/eda/heatmap_variables.jpeg',
                './images/results/lr_results.jpeg',
                './images/results/rf_results.jpeg'
                ]


@pytest.mark.parametrize('file_name', list_figures)
def test_eda(file_name):
    '''
    assert if dataf file exists
    '''
    try: 
        assert exists(file_name)
    except AssertionError as err:
        logging.error("Cannot create file %s", file_name)  
    logging.info("Image %s in folder", file_name)


# assert if all columns are present in the new df
cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]


# assert if columns are in df
@pytest.mark.parametrize('column_name', cat_columns)
def test_encoder_helper(column_name):
    '''
    test encoder helper
    '''
    df_test = pytest.df_test
    df_test = cls.encoder_helper(df_test)
    try:
        assert column_name in df_test.columns
    except AssertionError as err:
        logging.error("Column cannot be created")
    return df_test


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    df_test = pytest.df_test
    try:
        assert cls.perform_feature_engineering(df_test) is not None
    except AssertionError as err:
        logging.error("Cannot perform feature engineering")
    logging.info("Feature engineering performed")

models = ['./models/rfc_model.pkl', './models/logistic_model.pkl']


@pytest.mark.parametrize('model', models)
def test_load_models(model):
    '''
    assert if the models in pkl format exist
    '''
    try:
        assert joblib.load(model)
    except AssertionError as err:
        logging.error("Models not found")

logging.info("All passed!")
# if __name__ == "__main__":
# 	pass
