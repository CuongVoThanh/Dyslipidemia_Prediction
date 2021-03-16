import logging
import argparse
import torch

from lib.config import Config
from lib.runner import Runner

def parse_args():
    parser = argparse.ArgumentParser(description="Dyslipidemia Prediction")
    parser.add_argument("model", choices=["mlmodel", "dlmodel", "all"], 
                        help="ML Model, DL Model or Run all Models?")
    parser.add_argument("--mode", choices=["public", "private"], 
                        help="Using public test or private test?")
    parser.add_argument('--one_hot', default=True, type=bool, help='Encoding into one hot vector.')
    parser.add_argument('--drop_col', default=False, type=bool, help='Feature Selection Mode.')
    parser.add_argument('--kfold', default=5, type=int, help='Kfold Mode.')
    parser.add_argument('--pca_transform', default=300, type=int, help='PCA Mode. (Default 300 n_components)')
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning Rate (Default: 5e-4)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight_Decay (Default: 1e-5)")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs to test the model on (Default: 50)")
    parser.add_argument("--cpu", action="store_true", help="(Unsupported) Use CPU instead of GPU")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    device = torch.device('cpu') if not torch.cuda.is_available() or args.cpu else torch.device('cuda')
    cfg = Config(device, args.one_hot, args.drop_col, args.pca_transform, args.mode,
                    args.epochs, args.model, args.kfold, args.lr, args.weight_decay)

    if args.model in ['all', 'mlmodel', 'dlmodel']:
        try:
            runner = Runner(cfg)
            runner.run()
        except KeyboardInterrupt:
            logging.info('Building Model interrupted.')
    return

if __name__ == '__main__':
    main()