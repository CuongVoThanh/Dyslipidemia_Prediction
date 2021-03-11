import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
import numpy as np

TRAINING_PATH = ('./dataset/genotypes.csv', './dataset/treatment_outcome.csv')
SPLIT = 0.2
STATE = 42

def load_csv(data_path):
    """ Load CSV file from a path """
    return pd.read_csv(data_path)

def process_one_hot_data(df):
    """ Convert to one hot vector """
    # To categorical variable
    le = preprocessing.LabelEncoder()
    df = df.apply(le.fit_transform)
    
    # One hot encoding
    enc = preprocessing.OneHotEncoder()
    enc.fit(df)
    return enc.transform(df).toarray()

def load_train_val_set(split=SPLIT, is_one_hot=False, 
                    is_plot_data=False, is_drop_col=False, pca_transform=0):
    """
        Load and transform dataset by config mode 
        Return: 
            X, Y or X, Y divide into train/val set
    """
    
    X = load_csv(os.path.join(os.getcwd(), TRAINING_PATH[0]))
    Y = load_csv(os.path.join(os.getcwd(), TRAINING_PATH[1]))

    if is_plot_data:
        return X, Y

    if is_drop_col:
        with open('./dataset/drop_col.txt', 'r') as f:
            drop_cols = [line.replace("\\n", "").strip() for line in f.readlines()]
            f.close()
        X = X.drop(drop_cols, axis=1)

    if is_one_hot:
        X = process_one_hot_data(X)
        treatment = process_one_hot_data(Y[['treatment']])
        outcome = Y[['outcome']].to_numpy()
        X = np.hstack((X, treatment))
        Y = outcome
    
    if pca_transform:
        pca = PCA(n_components=pca_transform)
        pca.fit_transform(X)

    # Ramdom State 42 for reproducibility
    return train_test_split(X, Y, test_size=split, random_state=STATE, shuffle=True)