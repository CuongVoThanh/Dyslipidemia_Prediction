from lib.models.regression_models.linear_model import LinearModel
class Runner():
    def __init__(self, cfg):
        self.models = cfg.models
        self.data = cfg.dataset
    
    def run(self):
        for model, name in self.models:
            print(f'================== {name} ==================')
            linear_regression_model = LinearModel(*self.data, model)
            linear_regression_model.val()
