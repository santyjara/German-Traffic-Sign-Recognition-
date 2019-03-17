import numpy as np
import tensorflow as tf
import os
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv



class model:

    def __init__(self,input_shape,n_outputs,test_size,learning_rate,epochs=9,batch_size = 100):

        self.input_shape = input_shape
        self.n_outputs = n_outputs
        self.test_size = test_size
        self.learning_rate = learning_rate
        self.epochs =epochs
        self.batch_size = batch_size


    def load_data(self, folder_name):
        """

        :param folder_name: path of the folder that contains
        the subfolders named with the labels and their images
        :return: two arrays with the data stored the working
        directory ('x.npy' , 'y.npy')

        """
        y =[]
        X=[]

        for i in os.listdir(folder_name):
            boat_folder = os.path.join(folder_name, i)
            for j in (os.listdir(boat_folder)):

                label = int(boat_folder.split('/')[1][-2:])
                img=(io.imread(os.path.join(boat_folder, j)))
                image_resized = resize(img, (32, 32, 3),anti_aliasing=True)

                y.append(label)
                X.append(image_resized)


        X = np.array(X)
        y = np.array(y)

        np.save('x.npy',X)
        np.save('y.npy',y)

    def permute_feature_label_data(self):
        """Generates a random order and permutes the feature and label data accordingly."""

        feature_data = np.load('x.npy')
        label_data = np.load('y.npy')


        permutation = np.random.permutation(label_data.shape[0])
        # Reorganizes the given feature data and its labels in the permutation order.
        permuted_feature_data = feature_data[permutation, :, :, :]
        permuted_label_data = label_data[permutation]

        np.save('x.npy', permuted_feature_data)
        np.save('y.npy', permuted_label_data)

        return permuted_feature_data, permuted_label_data




    def split(self,X,y):

        """

        Normalize the images dividing them by 255 and split the data into train and test

        :param X: images (N,H,W,C)
        :param y: labels(N,)
        :param test_size: size of the test sample (0-1)
        :return: X_train, X_test, y_train, y_test according to the test size
        """

        #X = X - np.mean(X)
        X = X/255
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size)


        return X_train, X_test, y_train, y_test



    def train_model(self,X_train,y_train,X_test,y_test):
        """

        :param X_train:
        :param y_train:
        :param X_test:
        :param y_test:
        :return:
        """



        model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(filters=40, kernel_size=7, padding='same',activation=tf.nn.relu, input_shape=self.input_shape,kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=1),
                                    tf.keras.layers.Conv2D(filters=20,kernel_size=(5,5),padding='same',activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                                    #tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=1),
                                    #tf.keras.layers.Dropout(rate=0.3),
                                    tf.keras.layers.Conv2D(filters=10,kernel_size=(3,3),padding='same',activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                                    #tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=1),
                                    tf.keras.layers.Flatten(),
                                    #tf.keras.layers.Dropout(rate=0.2),
                                    tf.keras.layers.Dense(units=1024,activation=tf.nn.relu,use_bias=True),
                                    #tf.keras.layers.Dropout(rate=0.2),
                                    #tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(units=512,activation=tf.nn.relu,use_bias=True,kernel_regularizer=tf.keras.regularizers.l2(0.00)),
                                    tf.keras.layers.Dense(units=self.n_outputs,use_bias=True,activation=tf.nn.softmax)
                                    ])

        model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics =['accuracy'])

        model.fit(X_train, y_train,batch_size=self.batch_size,epochs=self.epochs)

        test_acc = model.evaluate(X_test, y_test)
        print('Test accuracy:', test_acc)

        model.save('my_model.h5')

    def load_data_to_predict(self,folder_name):

        X=[]

        with open('test.csv', 'r') as f:
            reader = csv.reader(f)
            your_list = list(reader)

        new_list = [your_list[i + 1][0] for i in range(len(your_list) - 1)]

        for i in (new_list):

            img=(io.imread(os.path.join(folder_name, i)))
            image_resized = resize(img, (32, 32, 3), anti_aliasing=True)
            image_resized = image_resized/255

            X.append(image_resized)

        images = np.array(X)

        return images,new_list

    def predict(self,X):

        """

        :param X: images (N,H,W,C)
        :return: predictions (N)

        """

        model = tf.keras.models.load_model('my_model.h5')

        predictions = model.predict(X)

        predictions = [np.argmax(predictions[i]) for i in range(predictions.shape[0])]


        return predictions

