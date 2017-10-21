#coding=utf-8
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.metrics import Accuracy
from utils import mnist_reader
import numpy as np

model_path = "./models/"
class Model_Alexnet(object):
	def __init__(self):
		# Building 'AlexNet'
        network = input_data(shape=[None, 28, 28, 1])
        network = conv_2d(network, 96, 11, strides=4, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = conv_2d(network, 256, 5, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = conv_2d(network, 384, 3, activation='relu')
        network = conv_2d(network, 384, 3, activation='relu')
        network = conv_2d(network, 256, 3, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = fully_connected(network, 4096, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 4096, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 10, activation='softmax')
        network = regression(network, metric=acc , optimizer='momentum',
                             loss='categorical_crossentropy',
                             to_one_hot = True , 
                             n_classes = 10 , 
                             learning_rate=0.001)
        self.model = tflearn.DNN(network, checkpoint_path='./models/fashion_alexnet',
                    max_checkpoints=1, tensorboard_verbose=2 , tensorboard_dir = "./logs/")

    def train(self , X_train , y_train , X_test , y_test ,n_epoch = 10 , path = model_path):
        if X_test is not None and y_test is not None:
            validation_set = (X_test , y_test)
        else:
            validation_set = 0.1
        self.model.fit(X_train, y_train, n_epoch=n_epoch, validation_set=validation_set, shuffle=True,
          show_metric=True, batch_size=40, snapshot_step=200,
          snapshot_epoch=False, run_id='alexnet_oxflowers17')
        self.model.save(path + "fashion_alexnet.tflearn")

    def predict(X_test):
        return self.model.predict(X_test)



class Model_Vgg16(object):
    def __init__(self):
        # Building 'VGG Network'
        network = input_data(shape=[None, 28 , 28 , 1])

        network = conv_2d(network, 64, 3, activation='relu')
        network = conv_2d(network, 64, 3, activation='relu')
        network = max_pool_2d(network, 2, strides=2)

        network = conv_2d(network, 128, 3, activation='relu')
        network = conv_2d(network, 128, 3, activation='relu')
        network = max_pool_2d(network, 2, strides=2)

        network = conv_2d(network, 256, 3, activation='relu')
        network = conv_2d(network, 256, 3, activation='relu')
        network = conv_2d(network, 256, 3, activation='relu')
        network = max_pool_2d(network, 2, strides=2)

        network = conv_2d(network, 512, 3, activation='relu')
        network = conv_2d(network, 512, 3, activation='relu')
        network = conv_2d(network, 512, 3, activation='relu')
        network = max_pool_2d(network, 2, strides=2)

        network = conv_2d(network, 512, 3, activation='relu')
        network = conv_2d(network, 512, 3, activation='relu')
        network = conv_2d(network, 512, 3, activation='relu')
        network = max_pool_2d(network, 2, strides=2)

        network = fully_connected(network, 4096, activation='relu')
        network = dropout(network, 0.5)
        network = fully_connected(network, 4096, activation='relu')
        network = dropout(network, 0.5)
        network = fully_connected(network, 10, activation='softmax')

        network = regression(network, optimizer='rmsprop',
                             loss='categorical_crossentropy',
                             to_one_hot = True , 
                             n_classes = 10 , 
                             learning_rate=0.0001)
        
        self.model = tflearn.DNN(network, checkpoint_path='./models/model_vgg',
                            max_checkpoints=1, tensorboard_verbose=2 , tensorboard_dir = "./logs/")

    def train(X_train , y_train , X_test = None , y_test = None , n_epoch = 10 , path = model_path):
        if X_test is not None and y_test is not None:
            validation_set = (X_test , y_test)
        else:
            validation_set = 0.1

        self.model.fit(X_train, y_train, validation_set = validation_set, n_epoch=n_epoch, shuffle=True,
          show_metric=True, batch_size=40, snapshot_step=500,
          snapshot_epoch=False, run_id='vgg_oxflowers17')
        self.model.save(path + "fashion_vgg.tflearn")

    def predict(X_test):
        return self.model.predict(X_test)

X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

X_train = X_train.reshape([-1 , 28 , 28 , 1])
X_test = X_test.reshape([-1 , 28 , 28 , 1])


def accuracy(y_pred , y_test):
    return sum(np.argmax(y_pred) == y_test) / (y_test.shape[0] * 1.0)
if __name__ == "__main__":
    alex = Model_Alexnet()
    vgg = Model_Vgg16()

    alex.train(X_train , y_train , X_test , y_test)
    y_pred_alex = alex.predict(X_test)
    vgg.train(X_train , y_train , X_test , y_test)
    y_pred_vgg = vgg.predict(X_test)

    print "Alex Acc: " , accuracy(y_pred_alex)
    print "VGG16 Acc: " , accuracy(y_pred_vgg)
    print "Bagging Acc: " , accuracy((y_pred_alex + y_pred_vgg) / 2)






