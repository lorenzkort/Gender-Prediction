# Import libraries
print('Importing libraries...', end=' ')
import time
start = time.time()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import svm
import pandas as pd

style.use('ggplot')
end = time.time()
print('took', round(end - start, 0), 'seconds')

def load_data():
    # importing data
    print('Importing data...', end=' ')
    start = time.time()
    df = pd.read_csv("Sample_data__sklearn_2.csv")
    global X
    X = df[[#'Binary'
            'Kilograms'
            ,'Centimeters']].values
    global y
    y = df.pop('_Gender').values
    end = time.time()
    print('took', round(end - start, 0), 'seconds') 

def train(c_score):
    # Creating SVC
    global clf
    clf = svm.SVC(kernel='linear', gamma='scale', C=c_score)

    # training data
    start = time.time()
    print('Training data...', end =" ")
    clf.fit(X, y)
    end = time.time()
    print('took', round(end - start, 0), 'seconds')

    # Predict input
    print('Predicting input...')
    pred = clf.predict([[78.76, 178.18]])
    print('Prediction: ', pred)

    # Show accuracy of algorithm
    ac = clf.score(X, y, sample_weight=None)
    print('Accuracy:', ac, 'with c:', c_score)
    return

def load_graph():
    # create plot of values
    w = clf.coef_[0] # Hyperplane co-efficient
    a = -w[0] / w[1]

    # Add line to plot
    xx = np.linspace(0, 150)
    yy = a * xx - clf.intercept_[0] / w[1]
    h = plt.plot(xx, yy, 'k-', label='Hyperplane')
    plt.scatter(X[:, 0], X[:, 1], c = y)
    plt.legend()
    plt.show()

# create list of c values
def c_list():
    r = [x for x in range(1,15)]
    return r

# Execute main function
def main():
    load_data()
    train(0.1)
    load_graph()

# Check if this Python file is the running file, then execute the main function
if __name__ == "__main__":
    main()