import os
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD
from sklearn.model_selection import KFold

#loading the dataset mnist
def load_dataset_mnist():
    #get the data
    (trainX, trainY), (testX, testY) = mnist.load_data()

    #make sure that the input images (28x28 px) have single channel
    trainX = trainX.reshape((trainX.shape[0],28,28,1))
    testX = testX.reshape((testX.shape[0],28,28,1))

    #make sure to have the output in the form that is easy for Neural Network to work with
    trainY = keras.utils.to_categorical(trainY)
    testY = keras.utils.to_categorical(testY)
    return trainX, trainY, testX, testY


#scale the pixels from (0,255) to (0,1)
def prep_pixels(train,test):
    train_ = train.astype('float32')
    test_ = test.astype('float32')

    train_ /= 255.
    test_ /= 255.

    return train_, test_


# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)
    scores, histories = list(), list()
	# prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
	# enumerate splits
    for train_ix, test_ix in kfold.split(dataX):
		# define model
        model = define_model()
		# select rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		# fit model
        history = model.fit(trainX, 
                      trainY, 
                      epochs=10, 
                      batch_size=32, 
                      validation_data=(testX, testY), 
                      verbose=2,
                      callbacks=[cp_callback])
		# evaluate model
        _, acc = model.evaluate(testX, testY, verbose=2)
        print('> %.3f' % (acc * 100.0))
		# stores scores
        scores.append(acc)
        histories.append(history)
    return scores, histories


'''# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=5*batch_size)'''


'''
# if the model is being run for the first time, only create the model and if we want to use the saved weights give first_time = 0
first_time = 1
if first_time:
    # Create a basic model instance
    model = create_model()
else:
    # Loads the weights
    model.load_weights(checkpoint_path)'''

     
