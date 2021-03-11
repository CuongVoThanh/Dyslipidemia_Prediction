# import yaml
# import torch

from sklearn.linear_model import (LinearRegression, Ridge, Lasso, 
                                ElasticNet, MultiTaskElasticNet)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.ensemble import (AdaBoostRegressor, ExtraTreesRegressor, 
                            GradientBoostingRegressor, RandomForestRegressor)

from utils.load_data import load_train_val_set
# from models.regression_models.nn_regression import NeuralNetwork

ML_MODELS = [LinearRegression(), Ridge(alpha=1.0), 
        Lasso(alpha=0.05), ElasticNet(), 
        MultiTaskElasticNet(alpha=0.1),
        SVR(C=1.0, epsilon=0.2),
        DecisionTreeRegressor(),
        xgb.XGBRegressor(verbosity=0),
        AdaBoostRegressor(random_state=0, n_estimators=100),
        ExtraTreesRegressor(),
        GradientBoostingRegressor(),
        RandomForestRegressor(max_depth=2, random_state=0)]

ML_MODEL_NAMES = ['Linear Regression',
        'Regularization L2 Linear Regression',
        'Regularization L1 Linear Regression',
        'ElasticNet',
        'MultiTask ElasticNet',
        'Support Vector Regression',
        'Decision Tree Regression',
        'XGBoost',
        'AdaBoost',
        'ExtraTreeRegressor',
        'GradientBoostingRegressor']

# DL_MODELS = [NeuralNetwork(input_shape)]
# DL_MODEL_NAMES = ['Neural Network']
class Config:
    def __init__(self, is_one_hot=True, is_drop_col=False, pca_transform=0, epochs=20, check_model='mlmodel'):
        self.ML_models = list(zip(ML_MODELS, ML_MODEL_NAMES))
        self.dataset = load_train_val_set(is_one_hot=is_one_hot, 
                                        is_drop_col=is_drop_col,
                                        pca_transform=pca_transform)
        self.epochs = epochs
        self.check_model = check_model
    # def load(self, path):
    #     with open(path, 'r') as file:
    #         self.config_str = file.read()
    #     self.config = yaml.load(self.config_str, Loader=yaml.FullLoader)

    def __getitem__(self, item):
        return self.config[item]

    def __contains__(self, item):
        return item in self.config
