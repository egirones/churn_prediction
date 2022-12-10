'''
author: Edgar Gironés
Date: 24.11.22
Churn operations for the dataset of bank,
including import, eda, encoding, model
building and prediction
'''

# import libraries
import os
import logging
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import dataframe_image as dfi
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_list import column_list
sns.set()

# load lists from asset folders
cat_columns = column_list.cat_columns
quant_columns = column_list.quant_columns
keep_cols = column_list.keep_cols

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename='./logs/results.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        dfx = pd.read_csv(pth)
        dfx['Churn'] = dfx['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        logging.info('File loaded')
        return df
    except FileNotFoundError:
        logging.error("Path to file: %s not found" % pth)
        raise FileNotFoundError(
            'File not found, need one valid path to continue')


def perform_eda(df_eda):
    '''
    perform eda on df and save figures to images folder
    input:
            df_eda: pandas dataframe
    output:
            None
    '''

    # for each column, check if it exists. If not, raise error
    plt.figure(figsize=(20, 10))
    df_eda['Churn'].hist()
    plt.title('Churn: Yes or No')
    plt.savefig('./images/churn.jpeg')

    plt.figure(figsize=(20, 10))
    df_eda['Customer_Age'].hist()
    plt.title('Age of customers')
    plt.savefig('./images/customer_age.jpeg')

    plt.figure(figsize=(20, 10))
    df_eda.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.title('Marital status')
    plt.savefig('./images/marital_status.jpeg')

    plt.figure(figsize=(20, 10))
    sns.histplot(df_eda['Total_Trans_Ct'], stat='density', kde=True)
    plt.title('Total Transactions')
    plt.savefig('./images/trans_ct.jpeg')

    plt.figure(figsize=(20, 10))
    sns.heatmap(df_eda.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.title('Heatmap of all variables')
    plt.savefig('./images/heatmap_variables.jpeg')


def encoder_helper(df_input, category_lst=cat_columns, response='Churn'):
    '''
    helper function to turn each categorical column
    into a new column with propotion of churn
    for each category

    input:
            df_input: pandas dataframe
            category_lst: list of columns that contain categorical
                features response: string of response
            name [optional argument that could be used
                for naming variables or index y column]
    output:
            df: pandas dataframe with new columns for
    '''
    for category in category_lst:
        try:
            lista = []
            groups = df.groupby(category).mean()['Churn']
            for val in df[category]:
                lista.append(groups.loc[val])
            new_column_name = f'{category}_{response}'
            df_input[new_column_name] = lista
        except KeyError:
            logging.error('Category %s not found' % category)
    logging.info('All categorical columns converted!')
    df_output = df_input
    return df_output


def perform_feature_engineering(
        df_input,
        response='Churn',
        columns_to_keep=keep_cols):
    '''
    input:
            df: pandas dataframe
            response: string of response name [optional
            argument that could be used for
            naming variables or index y column]
    output:
            x_train: X training data
            x_test: X testing data
            y_train: y training data
            y_test: y testing data
    '''
    X = df_input[columns_to_keep]
    y = df_input[response]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    logging.info('Train test split done')
    return x_train, x_test, y_train, y_test


def classification_report_image(real_values, predictions, name_figure):
    '''
    produces classification report for training and testing 
    results and stores report as image in images folder
    input:
            real_values: response values
            predictions:  predicted_values
            name_figure: name of figure
    output:
            None
    '''
    dfx = pd.DataFrame(
        classification_report(
            real_values,
            predictions,
            output_dict=True))
    name_figure_path = f'./images/{name_figure}.png'
    dfi.export(dfx, name_figure_path)
    logging.info(f'Image {name_figure_path} created')

def feature_importance_plot(
        importances,
        x_input,
        pth='./images/feature_importance.jpeg'):
    '''
    creates and stores the feature importances in pth
    input:
            importances: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
            None
    '''
    indices = np.argsort(importances)[::-1]
    names = [x_input.columns[i] for i in indices]
    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(x_input.shape[1]), importances[indices])
    plt.xticks(range(x_input.shape[1]), names, rotation=90)
    plt.savefig(pth)

def plot_lrc(
        x_test_input,
        y_test_input,
        lrc_path='./models/logistic_model.pkl',
        rf_path='./models/rfc_model.pkl',
        image_pth='./images/lrc_plot.jpeg'):
    '''
    return plot LRC
    input:
            x_test_input: data of X
            y_test_input:  data of Y
            lrc_path: LRC model path
            rf_path: RF model path
            image_pth: path where data is gonna be saved
    output:
            None
    '''
    rfc_model = joblib.load(rf_path)
    lr_model = joblib.load(lrc_path)

    lrc_plot = plot_roc_curve(lr_model, x_test_input, y_test_input)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(rfc_model, x_test_input, y_test_input, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(image_pth)
    logging.info(f'LRC plot saved at {image_pth}')


def plot_importances(x_columns, cv_rfc_pth='./models/rfc_model.pkl',
                     image_pth='./images/importances.jpeg'):
    '''
    plot importances
    input:
            X: df explanatory columngs
            cv_rfc_pth: path to the model
            pth: path to save image
    output:
        none
    '''
    cv_rfc = joblib.load(cv_rfc_pth)
    importances = cv_rfc.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [x_columns.columns[i] for i in indices]
    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(x_columns.shape[1]), importances[indices])
    plt.xticks(range(x_columns.shape[1]), names, rotation=90)
    plt.savefig(image_pth)
    logging.info(f'Importance plot saved at {image_pth}')


def train_models(x_train_input, x_test_input, y_train_input, y_test_input):
    '''
    train, store model results: images + scores, and store models
    input:
            x_train_input: X training data
            x_test_input: X testing data
            y_train_input: y training data
            y_test_input: y testing data
    output:
            None
    '''

    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train_input, y_train_input)

    lrc.fit(x_train_input, y_train_input)
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train_input)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test_input)
    logging.info('RFC done')

    y_train_preds_lr = lrc.predict(x_train_input)
    y_test_preds_lr = lrc.predict(x_test_input)
    logging.info('LRC done')

    # get classification images
    classification_report_image(y_test_input, y_test_preds_rf, 'test_results_rf')
    classification_report_image(y_train_input, y_train_preds_rf, 'train_results_rf')
    classification_report_image(y_test_input, y_test_preds_lr, 'test_results_lr')
    classification_report_image(y_train_input, y_train_preds_lr, 'test_results_lr')

    # save models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')
    logging.info('Models saved')

    # returns the plot
    plot_lrc(x_test_input, y_test_input)


if __name__ == "__main__":
    DATA_FILE = "./data/bank_data.csv"
    df = import_data(DATA_FILE)
    perform_eda(df)
    df = encoder_helper(df)
    x_train, x_test, y_train, y_test = perform_feature_engineering(df)
    train_models(x_train, x_test, y_train, y_test)
    plot_importances(df[keep_cols])
