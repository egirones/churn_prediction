'''
Churn operations for the dataset of bank, including import, eda, encoding, model building and prediction
'''

# import libraries
import os
import logging
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import dataframe_image as dfi


from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

os.environ['QT_QPA_PLATFORM']='offscreen'

cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'                
]

quant_columns = [
    'Customer_Age',
    'Dependent_count', 
    'Months_on_book',
    'Total_Relationship_Count', 
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 
    'Credit_Limit', 
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 
    'Total_Amt_Chng_Q4_Q1', 
    'Total_Trans_Amt',
    'Total_Trans_Ct', 
    'Total_Ct_Chng_Q4_Q1', 
    'Avg_Utilization_Ratio'
]

keep_cols = ['Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn', 
        'Income_Category_Churn',
        'Card_Category_Churn']

logging.basicConfig(filename='./logs/results.log',
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
                df = pd.read_csv(pth)
                df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
                logging.info('File loaded')
                return df
        except FileNotFoundError:
                logging.error(f'Path to file: {pth} not found')
                raise FileNotFoundError('File not found, need one valid path to continue')
        

def perform_eda(df):
        '''
        perform eda on df and save figures to images folder
        input:
                df: pandas dataframe

        output:
                None
        '''

        # for each column, check if it exists. If not, raise error 
        plt.figure(figsize=(20,10)) 
        df['Churn'].hist()
        plt.title('Churn: Yes or No')
        plt.savefig('./images/churn.jpeg');

        plt.figure(figsize=(20,10)) 
        df['Customer_Age'].hist()
        plt.title('Age of customers')
        plt.savefig('./images/customer_age.jpeg');

        plt.figure(figsize=(20,10)) 
        df.Marital_Status.value_counts('normalize').plot(kind='bar')
        plt.title('Marital status')
        plt.savefig('./images/marital_status.jpeg');

        plt.figure(figsize=(20,10)) 
        sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
        plt.title('Total Transactions')
        plt.savefig('./images/trans_ct.jpeg');
                        
        plt.figure(figsize=(20,10)) 
        sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
        plt.title('Heatmap of all variables')
        plt.savefig('./images/heatmap_variables.jpeg');

def encoder_helper(df, category_lst=cat_columns, response='Churn'):
        
        '''
        helper function to turn each categorical column into a new column with
        propotion of churn for each category - associated with cell 15 from the notebook

        input:
                df: pandas dataframe
                category_lst: list of columns that contain categorical features
                response: string of response name [optional argument that could be used for naming variables or index y column]

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
                        df[new_column_name] = lista
                except KeyError:
                        logging.error(f'Category {category} not found')
        logging.info('All categorical columns converted!')
        return df


def perform_feature_engineering(df, response='Churn', columns_to_keep=keep_cols):
        '''
        input:
                df: pandas dataframe
                response: string of response name [optional argument that could be used for naming variables or index y column]

        output:
                X_train: X training data
                X_test: X testing data
                y_train: y training data
                y_test: y testing data
        '''
        X = df[columns_to_keep]
        y = df[response]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)
        logging.info('Train test split done')
        return X_train, X_test, y_train, y_test

def classification_report_image(real_values, predictions, name_figure):
        '''
        produces classification report for training and testing results and stores report as image
        in images folder
        input:
                real_values: response values
                predictions:  predicted_values
                name_figure: name of figure
        output:
                None
        '''
        dfx = pd.DataFrame(classification_report(real_values, predictions, output_dict=True))
        name_figure_path = f'./images/{name_figure}.png'
        dfi.export(dfx, name_figure_path)
        logging.info(f'Image {name_figure_path} created')        
        return 

def feature_importance_plot(importances, X, pth='./images/feature_importance.jpeg'):
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
        names = [X.columns[i] for i in indices]
        plt.figure(figsize=(20,5))
        plt.title("Feature Importance")
        plt.ylabel('Importance')
        plt.bar(range(X.shape[1]), importances[indices])
        plt.xticks(range(X.shape[1]), names, rotation=90)
        plt.savefig(pth);


def plot_lrc(X_test, y_test, lrc_path='./models/logistic_model.pkl', rf_path='./models/rfc_model.pkl', image_pth='./images/lrc_plot.jpeg'):
        '''
        return plot LRC
        input: 
                X_test data of X
                y_test:  data of Y
                lrc_path: LRC model path
                rf_path: RF model path
                image_pth: path where data is gonna be saved
        output:
                none
        '''
        rfc_model = joblib.load(rf_path)
        lr_model = joblib.load(lrc_path)
        
        lrc_plot = plot_roc_curve(lr_model, X_test, y_test)
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        rfc_disp = plot_roc_curve(rfc_model, X_test, y_test, ax=ax, alpha=0.8)
        lrc_plot.plot(ax=ax, alpha=0.8)
        plt.savefig(image_pth);
        logging.info(f'LRC plot saved at {image_pth}')
        return

def plot_importances(X, cv_rfc_pth='./models/rfc_model.pkl', image_pth='./images/importances.jpeg'):
        '''
        plot importances
        input: 
                X: df explanatory columngs
                cv_rfc_pth: path to the model
                pth: path to save image
        '''
        cv_rfc = joblib.load(cv_rfc_pth)
        importances = cv_rfc.feature_importances_
        indices = np.argsort(importances)[::-1]
        names = [X.columns[i] for i in indices]
        plt.figure(figsize=(20,5))
        plt.title("Feature Importance")
        plt.ylabel('Importance')
        plt.bar(range(X.shape[1]), importances[indices])
        plt.xticks(range(X.shape[1]), names, rotation=90)
        plt.savefig(image_pth);
        logging.info(f'Importance plot saved at {image_pth}')
        return 


def train_models(X_train, X_test, y_train, y_test):
        '''
        train, store model results: images + scores, and store models
        input:
                X_train: X training data
                X_test: X testing data
                y_train: y training data
                y_test: y testing data
        output:
                None
        '''
        rfc = RandomForestClassifier(random_state=42)
        lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
        param_grid = { 
                'n_estimators': [200, 500],
                'max_features': ['auto', 'sqrt'],
                'max_depth' : [4,5,100],
                'criterion' :['gini', 'entropy']
        }
        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(X_train, y_train)
        
        lrc.fit(X_train, y_train)
        y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
        logging.info('RFC done')

        y_train_preds_lr = lrc.predict(X_train)
        y_test_preds_lr = lrc.predict(X_test)
        logging.info('LRC done')

        # get classification images 
        classification_report_image(y_test, y_test_preds_rf, 'test_results_rf')
        classification_report_image(y_train, y_train_preds_rf, 'train_results_rf')
        classification_report_image(y_test, y_test_preds_lr, 'test_results_lr')
        classification_report_image(y_train, y_train_preds_lr, 'test_results_lr')

        # save models
        joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
        joblib.dump(lrc, './models/logistic_model.pkl')
        logging.info('Models saved')

        # returns the plot
        plot_lrc(X_test, y_test)
        return

if __name__ == "__main__":
        data_file = "./data/bank_data.csv"
        df = import_data(data_file)
        perform_eda(df)
        df = encoder_helper(df)
        X_train, X_test, y_train, y_test = perform_feature_engineering(df)
        train_models(X_train, X_test, y_train, y_test)
        plot_importances(df[keep_cols])