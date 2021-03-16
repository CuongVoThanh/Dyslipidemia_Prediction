import torch
import numpy as np
import logging


class Runner():
    def __init__(self, cfg):
        self.cfg = cfg
        self.setup_logging()

    def run(self):
        """ Execute the model by config """
        if self.cfg.check_model == 'mlmodel' or self.cfg.check_model == 'all':
            self.logger.info("RUN MACHINE LEARNING MODELS")
            for model, name in self.cfg.ML_models:
                self.logger.info(f'Model: {name}')

                # Handling warming error
                if name in self.__get_ignore_warning_models():
                    self.cfg.data[2] = self.cfg.data[2].ravel()
                    self.cfg.data[3] = self.cfg.data[3].ravel()
                
                linear_regression_model = self.cfg.get_model('MLModel', *self.cfg.data, model, self.logger)
                train_score, eval_score = linear_regression_model.train()
                self.logger.info(f"[80-20] Training: {train_score:.3f} - Validation: {eval_score:.3f}")
                
                if self.cfg.kfold:
                    assert self.cfg.mode in ['public', 'private']

                    if self.cfg.mode == 'public':
                        X, y = self.merge_dataset(*self.cfg.data)
                    elif self.cfg.mode == 'private':
                        X, y = self.cfg.data[0], self.cfg.data[2]    

                    best_model = linear_regression_model.evaluate_model_by_kfold(X, y, linear_regression_model.model, self.cfg.kfold, self.logger)
                    self.logger.info('[{}-Folds] Validation: {:.3f}'.format(self.cfg.kfold, best_model.score))
                    
                    if self.cfg.mode == 'private':
                        test_score = best_model.predict(best_model.model, self.cfg.data[1], self.cfg.data[3])
                        self.logger.info('[{}-Folds] Test: {:.3f}'.format(self.cfg.kfold, test_score))

        if self.cfg.check_model == 'dlmodel' or self.cfg.check_model == 'all':
            self.logger.info("RUN DEEP LEARNING MODELS")
            self.logger.debug(f"DEVICE-IN-USE: {self.cfg.device}")
            
            X_train, X_val, y_train, y_val = [torch.from_numpy(data).float().to(self.cfg.device) for data in self.cfg.data]
            assert len(X_train) == len(y_train) and len(X_val) == len(y_val)

            for dlmodel_name in self.__get_dlmodel_names():
                self.logger.info(f'Model: {dlmodel_name}')
                dl_model = self.create_dlmodel_runner(dlmodel_name, X_train, X_val, y_train, y_val)
                
                # Train-Eval Part
                dl_model.train()
                dl_model.eval()

    def create_dlmodel_runner(self, name, X_train, X_val, y_train, y_val):
        """ Create deep learning model """
        assert name in self.__get_dlmodel_names()

        if name == 'ConvolutionalNeuralNetwork':
            X_train = torch.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1], -1))
            X_val = torch.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1], -1))
            self.cfg.model = self.cfg.get_model(name, X_train.shape[2])
        elif name == 'NeuralNetwork':
            self.cfg.model = self.cfg.get_model(name, X_train.shape[1])

        self.cfg.data = X_train, X_val, y_train, y_val
        self.cfg.device = self.cfg.device
        return self.cfg.get_model('DLModel', self.cfg, self.logger)

    def setup_logging(self):
        formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
        file_handler = logging.FileHandler(self.cfg.log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, stream_handler])
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def merge_dataset(X_train, X_val, y_train, y_val):
        return np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val])
    
    @staticmethod
    def __get_dlmodel_names():
        return [
            'NeuralNetwork',
            'ConvolutionalNeuralNetwork',
        ]

    @staticmethod
    def __get_ignore_warning_models():
        return [
            'Support Vector Regression',
            'AdaBoost',
            'ExtraTreeRegressor',
            'GradientBoostingRegressor',
            'LightGBM',
        ]