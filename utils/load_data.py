import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
import numpy as np

PUBLIC_DATASET_PATH = ['./dataset/genotypes.csv', './dataset/treatment_outcome.csv']
PRIVATE_DATASET_PATH = ['./dataset/private_test/data_train.csv','./dataset/private_test/label_train.csv',
                        './dataset/private_test/data_test.csv','./dataset/private_test/label_test.csv']
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
    enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
    enc.fit(df)
    return le, enc, enc.transform(df).toarray()

def drop_columns(df):
    with open('./dataset/drop_col.txt', 'r') as f:
        drop_cols = [line.replace("\\n", "").strip() for line in f.readlines()]
        f.close()
    return df.drop(drop_cols, axis=1)

def transform_by_pca(df, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(df)
    return pca, pca.transform(df)

def load_train_val_public_set(split=SPLIT, is_one_hot=False, 
                    is_plot_data=False, is_drop_col=False, pca_transform=0):
    """
        Load and transform public dataset by config mode 
        Return: 
            X, Y or X, Y divide into train/val set
    """
    
    X = load_csv(os.path.join(os.getcwd(), PUBLIC_DATASET_PATH[0]))
    Y = load_csv(os.path.join(os.getcwd(), PUBLIC_DATASET_PATH[1]))

    if is_plot_data:
        return X, Y

    if is_drop_col:
        X = drop_columns(X)

    if is_one_hot:
        _, _, X = process_one_hot_data(X)
        _, _, treatment = process_one_hot_data(Y[['treatment']])
        outcome = Y[['outcome']].to_numpy()
        X = np.hstack((X, treatment))
        Y = outcome
    
    if pca_transform:
        _, X = transform_by_pca(X, pca_transform)

    # Ramdom State 42 for reproducibility
    return train_test_split(X, Y, test_size=split, random_state=STATE, shuffle=True)

def load_train_val_private_set(is_one_hot=False, is_drop_col=False, pca_transform=0):
    """
        Load and transform private dataset by config mode 
        Return: 
            X_train, X_test, y_train, y_test
    """
    X_train = load_csv(os.path.join(os.getcwd(), PRIVATE_DATASET_PATH[0]))
    y_train = load_csv(os.path.join(os.getcwd(), PRIVATE_DATASET_PATH[1]))
    X_test = load_csv(os.path.join(os.getcwd(), PRIVATE_DATASET_PATH[2]))
    y_test = load_csv(os.path.join(os.getcwd(), PRIVATE_DATASET_PATH[3]))

    if is_drop_col:
        X_train = drop_columns(X_train)
        X_test = drop_columns(X_test)

    if is_one_hot:
        le, enc, X_train = process_one_hot_data(X_train)
        X_test = enc.transform(X_test.apply(le.fit_transform)).toarray()
    
    if pca_transform:
        pca, X_train = transform_by_pca(X_train, pca_transform)
        X_test = pca.transform(X_test)
   
    return [X_train, X_test, y_train.to_numpy(), y_test.to_numpy()]