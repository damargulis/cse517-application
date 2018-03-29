import csv
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.model_selection import cross_val_score

DATA_FILE = '../Data/OnlineNewsPopularity.csv'

def get_dataset():
    with open(DATA_FILE, 'r') as data_file:
        reader = csv.reader(data_file)
        dataset = [line for line in reader]
        for i in range(len(dataset)):
            for j in range(len(dataset[i])):
                try:
                    dataset[i][j] = float(dataset[i][j])
                except ValueError:
                    pass
        return dataset[0], dataset[1:]

def xy_split(dataset):
    train, test = train_test_split(dataset, test_size=.2)
    x_train = [point[2:-1] for point in train]
    y_train = [point[-1] for point in train]
    x_test = [point[2:-1] for point in test]
    y_test = [point[-1] for point in test]
    return x_train, y_train, x_test, y_test

def main():
    print('getting dataset')
    attributes, dataset = get_dataset()
    print('spliting')
    x_train, y_train, x_test, y_test = xy_split(dataset)
    print('creating models')
    model1 = GaussianProcessRegressor(kernel=ConstantKernel())
    model2 = GaussianProcessRegressor(kernel=RBF())

    print('cross valing 1')
    scores1 = cross_val_score(model1, x_train, y_train, cv=10)
    print('cross valing 2')
    scores2 = cross_val_score(model2, x_train, y_train, cv=10)

    print(scores1)
    print(scores2)

if __name__ == '__main__':
    main()
