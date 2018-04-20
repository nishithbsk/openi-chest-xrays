import sys
import numpy as np

from cnn import CNN
from utils import load_X_and_Y

def train():
    # load data
    X, Y = load_X_and_Y()
    x_train, x_dev, x_test = X
    y_train, y_dev, y_test = Y

    # train model
    model = CNN()
    model.fit(x_train, y_train, x_dev, y_dev, save=True)
    model.evaluate(x_test, y_test)

if __name__ == "__main__":
    train()
