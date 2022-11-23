'''
Module for the logging and testing of the churn prediction 
'''
import os
from os.path import exists
import logging
import churn_library as cls
import pytest
import joblib


logging.basicConfig(
    level = logging.DEBUG,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import():
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = cls.import_data("./data/bank_data.csv")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err
	pytest.df = df
	return df

list_figures = ['./images/churn.jpeg',
					'./images/customer_age.jpeg',
					'./images/marital_status.jpeg',
					'./images/trans_ct.jpeg',
					'./images/heatmap_variables.jpeg'
					]

@pytest.mark.parametrize('file_name', list_figures)
def test_eda(file_name):
	assert exists(file_name)

# assert if all columns are present in the new df
cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'                
]
@pytest.mark.parametrize('column_name', cat_columns)
def test_encoder_helper(column_name):
	'''
	test encoder helper
	'''
	df = pytest.df
	df = cls.encoder_helper(df)
	assert column_name in df.columns
	pytest.df = df
	return df

def test_perform_feature_engineering():
	'''
	test perform_feature_engineering
	'''
	df = pytest.df
	assert cls.perform_feature_engineering(df) != None

list_figures_test = ['test_results_rf',
				'train_results_rf',
				'test_results_lr',
				'test_results_lr'
				]

@pytest.mark.parametrize('figure', list_figures_test)
def test_train_models(figure):
	'''
	test train_models
	'''
	assert exists(f'./images/{figure}.png')

models = ['./models/rfc_model.pkl', './models/logistic_model.pkl']

@pytest.mark.parametrize('model', models)
def test_load_models(model):
	assert joblib.load(model)

# if __name__ == "__main__":
# 	pass








