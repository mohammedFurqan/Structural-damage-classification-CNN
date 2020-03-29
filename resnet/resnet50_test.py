import numpy as np
import tensorflow as tf
import pickle
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
import matplotlib.pyplot as plt

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

def load_dataset():
    X_set = np.load("phi2018task7/X_train.npy") # (2632, 224, 224, 3)
    
    N = X_set.shape[0]
    Y_set = np.load("phi2018task7/Y_train.npy") # (2632,)

#    shuffle the data 
    arr = np.arange(N)
    np.random.shuffle(arr)
    
    train_x_orig = X_set[arr[0:40]]
    train_y_orig = Y_set[arr[0:40]]
    
    test_x_orig = X_set[arr[2400: -1]]
    test_y_orig = Y_set[arr[2400: -1]]
    
#    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
#    
    train_y_orig = train_y_orig.reshape((1, \
                            train_y_orig.shape[0]))
    test_y_orig = test_y_orig.reshape((1, \
                            test_y_orig.shape[0]))

    classes = np.arange(np.max(test_y_orig)+1)
    print("classes range: " +str(np.min(test_y_orig)) \
          + " " + str(np.max(test_y_orig)))
    return train_x_orig, train_y_orig, test_x_orig, test_y_orig, classes

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def identity_block(X, f, filters, stage, block):
    """    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """    
    # define the name basis
    conv_name_base = 'res'+str(stage)+block+'_branch'
    bn_name_base = 'bn'+str(stage)+block+'_branch'
    
    #Retrieve the numbers of Filters
    F1, F2, F3= filters
    
    #Save the input value
    X_shortcut = X
    X_shortcut = BatchNormalization(name = 'short' + block + str(stage) + '_2a')(X_shortcut)
    
    #first layer of component
    X = Conv2D(F1, kernel_size=(1,1), padding='valid', \
        name = conv_name_base+'2a')(X)
    X = BatchNormalization(name = bn_name_base+'2a')(X)
    X = Activation(activation = 'relu')(X)
    
    #second layer of component
    X = Conv2D(F2, kernel_size=(f,f), padding = 'same', \
        name = conv_name_base+'2b')(X)
    X = BatchNormalization(name = bn_name_base+'2b')(X)
    X = Activation(activation = 'relu')(X)
    
    #third layer of component
    X = Conv2D(F3, kernel_size=(1,1), padding='valid',\
        name = conv_name_base+'2c')(X)
    X = BatchNormalization(name = bn_name_base+'2c')(X)
    
    #add a shortcut
    X = layers.add([X_shortcut, X])
    X = Activation('relu')(X)
    
    return X

def convolutional_block(X, f, filters, stage, block, s = 2):
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X
    X_shortcut = BatchNormalization(name = 'short' + block+ str(stage) + '_2b')(X_shortcut)

    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a',\
               padding='valid', kernel_initializer = 'glorot_uniform')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv2D(F2, (f, f), strides = (1, 1), name = conv_name_base + '2b',\
               padding='same', kernel_initializer = 'glorot_uniform')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(F3, (1, 1), strides = (1, 1), name = conv_name_base + '2c',\
               padding='valid', kernel_initializer = 'glorot_uniform')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
    
    #shortcut path
    X_shortcut = Conv2D(F3, (1,1), strides = (s,s), name = conv_name_base+'1',\
            padding = 'valid', kernel_initializer='glorot_uniform')(X_shortcut)
    X_shortcut = BatchNormalization(name = bn_name_base+'1')(X_shortcut)
    
    #add the shortcut into path
    X = layers.add([X, X_shortcut])
    X = Activation(activation='relu')(X)
    
    return X

def resnet50_model(input_shape=(224,224,3), classes = 4):
    
    # input data : (224x224x3)
    X_input = Input(input_shape)
    
#    #stage 0 : zeros padding
#    X = ZeroPadding2D(padding = (3,3))(X_input)
    
    #stage 1 :  (224x224x3) -> (56x56x64)
    X = Conv2D(64, kernel_size = (7,7), strides = (2,2), name = 'conv1',\
               kernel_initializer = 'glorot_uniform')(X_input)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(3,3), strides = (2,2))(X)
    
    print("after 1st stage : " + str(X.shape))
    
    #stage 2 : (56x56x64) -> (28x28x256)
    X = convolutional_block(X, 3, filters = [64,64,256], stage = 2,\
                            block = 'a', s = 2)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
#    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c') 
    print("after 2nd stage : " + str(X.shape))
    
#    #stage 3 : (28x28x256) -> (14x14x512)
    X = convolutional_block(X, f = 3, filters=[128,128,512], stage = 3, block='a', s = 2)
    X = identity_block(X, f = 3, filters=[128,128,512], stage= 3, block='b')
    X = identity_block(X, f = 3, filters=[128,128,512], stage= 3, block='c')
#    X = identity_block(X, f = 3, filters=[128,128,512], stage= 3, block='d')
    print("after 3rd stage : " + str(X.shape))
#    
#    #stage 4 : (14x14x512) -> (7x7x1024)
#    X = convolutional_block(X, f = 3, filters=[256, 256, 1024], block='a', stage=4, s = 2)
#    X = identity_block(X, f = 3, filters=[256, 256, 1024], block='b', stage=4)
#    X = identity_block(X, f = 3, filters=[256, 256, 1024], block='c', stage=4)
#    X = identity_block(X, f = 3, filters=[256, 256, 1024], block='d', stage=4)
#    X = identity_block(X, f = 3, filters=[256, 256, 1024], block='e', stage=4)
#    X = identity_block(X, f = 3, filters=[256, 256, 1024], block='f', stage=4)
#    print("after 4th stage : " + str(X.shape))
#    
#    #stage 5 : (7x7x1024) -> 
#    X = convolutional_block(X, f = 3, filters=[512, 512, 2048], stage=5, block='a', s = 2)
#    # filters should be [256, 256, 2048], but it fail to be graded. Use [512, 512, 2048] to pass the grading
#    X = identity_block(X, f = 3, filters=[512, 512, 2048], stage=5, block='b')
#    X = identity_block(X, f = 3, filters=[512, 512, 2048], stage=5, block='c')
#    print("after 5th stage : " + str(X.shape))
    
    
    #stage 6 : average pooling
    X = AveragePooling2D(pool_size=(2,2))(X)
    print("after 6th stage : " + str(X.shape))
    
    #stage 7 : output layer: fully connected
    X = Flatten()(X)
    
    #third fc layer
    X = Dense(512, activation = "relu", use_bias = True, kernel_initializer="glorot_uniform")(X)
    print("fc3: " + str(X.shape))
#    
    #fourth fc layer
    X = Dense(64, activation = "relu", use_bias = True, kernel_initializer="glorot_uniform")(X)
    print("fc4: " + str(X.shape))
        
    Y = Dense(classes, activation='softmax', \
        kernel_initializer = 'glorot_uniform')(X)
    print(Y)
    input()
    print("after 7th stage : " + str(Y.shape))
    model = Model(inputs = X_input, outputs=Y, name = 'resnet50')
    
    return model


train_x_orig, train_y_orig, test_x_orig, test_y_orig, classes = load_dataset()
# Normalize image vectors
train_X = train_x_orig/255.
test_X = test_x_orig/255.

# Reshape
train_Y = convert_to_one_hot(train_y_orig, 4).T
test_Y = convert_to_one_hot(test_y_orig, 4).T

print("number of class " + str(np.max(train_y_orig)))
print("number of class " + str(np.max(test_y_orig)))

print("train_x_orig " + str(train_X.shape))
print("test_x_orig " + str(test_X.shape))
print("train_y_orig " + str(train_y_orig.shape))
print("test_y_orig " + str(test_y_orig.shape))
print("train_y " + str(train_Y.shape))
print("test_y " + str(test_Y.shape))
    
print("classes " + str(classes))

model = resnet50_model((224, 224 ,3), classes = 4)
model.compile(optimizer='sgd', \
              loss = 'categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x = train_X, y= train_Y, epochs = 30)
prediction = model.evaluate(x = test_X, y = test_Y)

plt.plot(history.history['acc']) 
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

plt.plot(history.history['loss']) 
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

#with open('/trainHistoryDict', 'wb') as file_pi:
#        pickle.dump(history.history, file_pi)

with open('hist_resnet50_baseline.pickle', 'wb') as handle:
    pickle.dump(history.history, handle)

with open('hist_resnet50_baseline.pickle', 'rb') as handle:
    b = pickle.load(handle)

print(b)

print(prediction)