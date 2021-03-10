import logging
import argparse

import warnings
warnings.filterwarnings("ignore")

from lib.config import Config
from lib.runner import Runner

def parse_args():
    parser = argparse.ArgumentParser(description="Dyslipidemia Prediction")
    parser.add_argument("model", choices=["mlmodel", "dlmodel", "all"], 
                        help="ML Model, DL Model or Run all Models?")

    parser.add_argument('--one_hot', default=True, type=bool, help='Encoding into one hot vector.')
    parser.add_argument('--drop_col', default=False, type=bool, help='Feature Selection Mode.')
    parser.add_argument('--pca_transform', default=300, type=int, help='PCA Mode. (default 300 n_components)')
    parser.add_argument("--epochs", type=int, help="Epochs to test the model on (Default: 20)")
    
    #TODO: Move all soft parameter to yml config
    # parser.add_argument("--cfg", help="Config file")

    # parser.add_argument("--cpu", action="store_true", help="(Unsupported) Use CPU instead of GPU")

    # parser.add_argument("--save_predictions", action="store_true", help="Save predictions to pickle file")
    # parser.add_argument("--view", choices=["all", "mistakes"], help="Show predictions")

    args = parser.parse_args()
    if args.model not in ["mlmodel", "dlmodel", "all"]:
        raise Exception("Please input mode in these options: mlmodel, dlmodel, all")

    return args

def main():
    args = parse_args()
    cfg = Config(args.one_hot, args.drop_col, args.pca_transform, args.epochs)
    
    # for pytorch implementation
    # device = torch.device('cpu') if not torch.cuda.is_available() or args.cpu else torch.device('cuda')

    #TODO: Refactor code
    if args.model in ['all', 'mlmodel', 'dlmodel']:
        try:
            if args.model == 'dlmodel':
                runner = Runner(cfg,check_ML=False)
            else:
                runner = Runner(cfg,check_ML=True)
            runner.run()
        except KeyboardInterrupt:
            logging.info('Building Model interrupted.')
    return

if __name__ == '__main__':
    main()

