import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

img_dims = (224, 224, 3)

def show_xray(xray):
    plt.imshow(xray)
    plt.show()

def preprocess_xray(xray_path):
    xray = image.load_img(xray_path, grayscale=True,
                          target_size=(img_dims[0], img_dims[1], 1))
    xray = image.img_to_array(xray)
    xray = np.dstack([xray, xray, xray])
    xray = preprocess_input(xray)
    return xray

def load_X_and_Y(num_examples=-1):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataset = np.load(os.path.join(curr_dir, "../data/dataset.npy"))

    x_train, x_dev, x_test = [], [], []
    y_train, y_dev, y_test = [], [], []

    if num_examples != -1:
      dataset = dataset[:num_examples]

    last_train_idx = int(0.8 * len(dataset))
    for idx, entry in enumerate(tqdm(dataset)):
        xray_path = entry["xray_paths"][0]
        xray = preprocess_xray(xray_path)
        label = float(entry["label"])

        if idx <= last_train_idx:
            x_train.append(xray)
            y_train.append(label)
        else:
            if idx % 2 == 0:
                x_dev.append(xray)
                y_dev.append(label)
            else:
                x_test.append(xray)
                y_test.append(label)

    X = [np.array(x_train), np.array(x_dev), np.array(x_test)]
    y_train = np.array(y_train).reshape((-1, 1))
    y_dev = np.array(y_dev).reshape((-1, 1))
    y_test = np.array(y_test).reshape((-1, 1))
    Y = [y_train, y_dev, y_test]

    print "Number of training examples:", len(x_train)
    print "Number of dev examples:", len(x_dev)
    print "Number of test examples:", len(x_test)

    return X, Y
