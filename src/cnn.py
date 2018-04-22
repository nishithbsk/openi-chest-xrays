import keras
import numpy as np

from datetime import datetime
from keras import optimizers
from keras.models import Model, load_model
from keras.layers import Dense
from keras.applications.resnet50 import ResNet50
from sklearn.metrics import roc_auc_score

from utils import img_dims

batch_size = 64
num_epochs = 15
lr = 0.0001
lr_decay = 0.005

class CNN(object):
    def __init__(self,
                 img_dims=img_dims,
                 lr=lr, lr_decay=lr_decay,
                 save_name='CNN_'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')):
        self._save_name = save_name
        
        model = ResNet50(include_top=False, weights='imagenet',
                         pooling='max', input_shape=img_dims)
        pred = Dense(1, activation='sigmoid')(model.layers[-1].output)
        self._model = Model(input=model.input, output=pred)
        self._model.compile(loss=keras.losses.binary_crossentropy,
            optimizer=keras.optimizers.Adam(lr=lr, decay=lr_decay),
            metrics=['accuracy'])

    def save(self):
        self._model.save(self._save_name + '.h5')

    def fit(self,
            x_train, y_train,
            x_dev, y_dev,
            batch_size=batch_size,
            num_epochs=num_epochs,
            save=True):
        self._model.fit(x_train, y_train,
                        validation_data=(x_dev, y_dev),
                        batch_size=batch_size,
                        epochs=num_epochs)
        if save:
            self.save()

    def predict(self, x):
        return self._model.predict(x)

    def evaluate(self, x_test, y_test):
        preds = self.predict(x_test).reshape((-1, 1))
        roc_auc = roc_auc_score(y_test, preds)
        return roc_auc

