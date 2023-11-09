import pandas as pd
import numpy as np
from sklearn import preprocessing


TEST_FILE = './data/test.csv'
TRAIN_FILE = './data/train.csv'

TAG = ['year', 'month', 'day', 'week', 'day_2', 'day_1', 'average', 'actual']

def _load(file):
    features = pd.read_csv(file, names=TAG, header=0)
    # 获取GT
    all_labels = np.array(features['actual'])
    # 在特征中去除GT
    features = features.drop('actual', axis=1)
    features = pd.get_dummies(features)

    all_features = preprocessing.StandardScaler().fit_transform(features)

    return all_features, all_labels


def load_train():
    return _load(TRAIN_FILE)

def load_test():
    return _load(TEST_FILE)

if __name__ == '__main__':
    # for TESTs
    all_features_train, all_labels_train = load_train()
    all_features_test, all_labels_test = load_test()

    print(all_features_train.shape)
    print(all_labels_train.shape)

    # print(all_features_train)
    # print(all_labels_train)
    
