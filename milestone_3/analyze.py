import csv
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

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

def xy_split(dataset, test_size=0):
    train, test = train_test_split(dataset, test_size=test_size)
    x_train = [point[2:-1] for point in train]
    y_train = [point[-1] for point in train]
    x_test = [point[2:-1] for point in test]
    y_test = [point[-1] for point in test]
    return x_train, y_train, x_test, y_test

def xy_split_new(dataset, test_size):
    train, test = train_test_split(dataset)
    x_train = [point[:-1] for point in train]
    y_train = [point[-1] for point in train]
    x_test = [point[:-1] for point in test]
    y_test = [point[-1] for point in test]
    return x_train, y_train, x_test, y_test

def analyze(x, y):
    dataset = []
    for i in range(len(x)):
        dataset.append([x[i][0], x[i][1], y[i]])
    print('1')
    x_train, y_train, x_test, y_test = xy_split_new(dataset, .2)
    print('2')
    regr = linear_model.LinearRegression()
    print('3')
    print(len(x_train))
    print(len(y_train))
    import pdb; pdb; pdb.set_trace()
    scores = cross_val_score(regr, x_train, y_train, cv=10)
    print('4')
    regr.fit(x_train, y_train)
    print('5')
    predictions = regr.predict(x_test)
    with open('pca_linear.csv', 'w') as output:
        writer = csv.writer(output)
        writer.writerow([''])
        writer.writerow(['Ran Linear Regression'])
        writer.writerow(['Cross Val Scores'])
        writer.writerow(scores)
        writer.writerow(['Coefficients: '])
        writer.writerow(regr.coef_)
        writer.writerow(['Mean squared error: '])
        writer.writerow([mean_squared_error(y_test, predictions)])
        writer.writerow(['Variance score:'])
        writer.writerow([r2_score(y_test, predictions)])

def main():
    print("Starting")
    print("Getting Dataset")
    attributes, dataset = get_dataset()
    print("Spliting")
    x_train, y_train, _, _ = xy_split(dataset)
    print("Creating PCA")
    pca = PCA(n_components=2)
    print("Fiting")
    pca.fit(x_train)
    print("Transforming")
    transformed = pca.transform(x_train)

    analyze(transformed, y_train)

    print("Creating plot")
    n_points = len(x_train)
    THRESHOLD = 500
    viral = []
    not_viral = []
    for i in range(n_points):
        if y_train[i] >= THRESHOLD:
            viral.append(transformed[i])
        else:
            not_viral.append(transformed[i])
    viral = np.array(viral)
    not_viral = np.array(not_viral)
    plt.figure()
    plt.scatter(
            viral[:, 0],
            viral[:, 1],
            color="blue",
            alpha=.8,
            lw=2,
            label="viral"
    )
    plt.scatter(
            not_viral[:, 0],
            not_viral[:, 1],
            color="green",
            alpha=.8,
            lw=2,
            label="not viral"
    )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title('PCA')
    plt.show()

if __name__ == "__main__":
    main()

