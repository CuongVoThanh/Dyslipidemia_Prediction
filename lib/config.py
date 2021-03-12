# import yaml

from sklearn.linear_model import (LinearRegression, Ridge, Lasso, 
                                ElasticNet, MultiTaskElasticNet)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.ensemble import (AdaBoostRegressor, ExtraTreesRegressor, 
                            GradientBoostingRegressor, RandomForestRegressor)
from lightgbm import LGBMRegressor

from utils.load_data import load_train_val_set


class Config:
    def __init__(self, device, is_one_hot=True, is_drop_col=False, pca_transform=0, 
                    epochs=20, check_model='mlmodel', kfold=0):
        self.ML_models = self.__get_ML_models()
        self.dataset = load_train_val_set(is_one_hot=is_one_hot, 
                                        is_drop_col=is_drop_col,
                                        pca_transform=pca_transform)
        self.epochs = epochs
        self.check_model = check_model
        self.device = device
        self.kfold = kfold

    # def load(self, path):
    #     with open(path, 'r') as file:
    #         self.config_str = file.read()
    #     self.config = yaml.load(self.config_str, Loader=yaml.FullLoader)
    
    @staticmethod
    def __get_ML_models():
        return [
            (LinearRegression(), 'Linear Regression'),
            (Ridge(alpha=1.0), 'Regularization L2 Linear Regression'),
            (Lasso(alpha=0.05), 'Regularization L1 Linear Regression'),
            (ElasticNet(), 'ElasticNet'),
            (MultiTaskElasticNet(alpha=0.1), 'MultiTask ElasticNet'),
            (SVR(C=1.0, epsilon=0.2), 'Support Vector Regression'),
            (DecisionTreeRegressor(), 'Decision Tree Regression'),
            (xgb.XGBRegressor(verbosity=0), 'XGBoost'),
            (AdaBoostRegressor(random_state=0, n_estimators=100), 'AdaBoost'),
            (ExtraTreesRegressor(), 'ExtraTreeRegressor'),
            (GradientBoostingRegressor(), 'GradientBoostingRegressor'),
            (RandomForestRegressor(max_depth=2, random_state=0), 'RandomForest'),
            (LGBMRegressor(), 'LightGBM')
        ]

    def __getitem__(self, item):
        return self.config[item]

    def __contains__(self, item):
        return item in self.config
