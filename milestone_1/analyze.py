import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
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

def describe(attributes, dataset, output_file):
    with open(output_file, 'w') as output:
        writer = csv.writer(output)
        writer.writerow(['Num Attributes: ' + str(len(attributes))])
        writer.writerow(['Num Instances: ' + str(len(dataset))])

        writer.writerow(['Summary Statistics:'])
        writer.writerow([
            'Feature', 
            'Min', 
            'Max', 
            'Avg', 
            'Std',
            'Varience',
        ])
        for i, attribute in enumerate(attributes):
            if i < 2:
                continue
            attribute_data = [article[i] for article in dataset]
            writer.writerow([
                attribute,
                min(attribute_data),
                max(attribute_data),
                np.mean(attribute_data),
                np.std(attribute_data),
                np.var(attribute_data),
            ])

def xy_split(dataset):
    train, test = train_test_split(dataset, test_size=.2)
    x_train = [point[2:-1] for point in train]
    y_train = [point[-1] for point in train]
    x_test = [point[2:-1] for point in test]
    y_test = [point[-1] for point in test]
    return x_train, y_train, x_test, y_test

def main():
    attributes, dataset = get_dataset()
    describe(attributes, dataset, 'description.csv')
    x_train, y_train, x_test, y_test = xy_split(dataset)
    regr = linear_model.LinearRegression()
    scores = cross_val_score(regr, x_train, y_train, cv=10)
    regr.fit(x_train, y_train)
    predictions = regr.predict(x_test)
    with open('linear.csv', 'w') as output:
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


if __name__ == '__main__':
    main()
