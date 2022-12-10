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


logging.basicConfig(
    level=logging.DEBUG,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import - this example is completed for
    you to assist with the other test functions
    '''
    try:
        df_test = cls.import_data("./data/bank_data.csv")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df_test.shape[0] > 0
        assert df_test.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    pytest.df_test = df_test
    return df_test


list_figures = ['./images/churn.jpeg',
                './images/customer_age.jpeg',
                './images/marital_status.jpeg',
                './images/trans_ct.jpeg',
                './images/heatmap_variables.jpeg',
                './images/lr_results.jpeg',
                './images/rf_results.jpeg'
                ]


@pytest.mark.parametrize('file_name', list_figures)
def test_eda(file_name):
    '''
    assert if dataf file exists
    '''
    assert exists(file_name)


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
    assert column_name in df_test.columns
    
    return df_test


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    df_test = pytest.df_test
    assert cls.perform_feature_engineering(df_test) is not None
    
models = ['./models/rfc_model.pkl', './models/logistic_model.pkl']


@pytest.mark.parametrize('model', models)
def test_load_models(model):
    '''
    assert if the models in pkl format exist
    '''
    assert joblib.load(model)

# if __name__ == "__main__":
# 	pass
