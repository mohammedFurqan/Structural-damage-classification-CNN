import tensorflow as tf
import keras
from keras.layers import Input, Dense, Activation, ZeroPadding2D,\
BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout,\
GlobalMaxPool2D, GlobalAveragePooling2D
from keras.models import Model
import keras.backend as K
K.set_image_data_format("channels_last")

import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_dataset():
    X_set = np.load("../data_phi2018/task7_X_train.npy") # (2632, 224, 224, 3)
    
    N = X_set.shape[0]
    Y_set = np.load("../data_phi2018/task7_y_train.npy") # (2632,)

#    shuffle the data 
    arr = np.arange(N)
    np.random.shuffle(arr)
    
    train_x_orig = X_set[arr[0:2400]]
    train_y_orig = Y_set[arr[0:2400]]
    
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

def vgg16_model(input_shape):
    """
    Arguments:
        input_shape: the shape of the image of datasets
        
    Returns:
        model:a Model() instance in Keras
    """
    # Define the input placeholder as a tensor with shape input_shape. 
    # Think of this as your input image!
    X_input = Input(input_shape)
    
    print("initial: " + str(X_input.shape))
    
    #first layer
    #X = Dropout(0,2)(X_input)
#    X = Conv2D(32, kernel_size=(3,3), padding='same', strides=(1,1), name = 'conv11')(X_input)
#    print("conv11:" +str(X.shape))
    X = Conv2D(64, kernel_size=(11,11), padding='valid', strides=(4,4), name = 'conv12')(X_input)
    print("conv12:" +str(X.shape))
    X = BatchNormalization()(X)
    X = Activation('relu')(X)    
    #first layer followed by a max pooling layer
    X = MaxPooling2D(pool_size=(2,2), padding='valid', strides=(2,2), name = 'max1')(X)
    print("maxpool1: " + str(X.shape))

    #second layer
    X = Conv2D(64, kernel_size=(5,5), padding='same', strides=(1,1), name = 'conv21')(X)
    print("conv21:" +str(X.shape))
    X = Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), name = 'conv22')(X)
    print("conv22:" +str(X.shape))
    X = BatchNormalization()(X)
    X = Activation('relu')(X)    
    #second layer followed by a max pooling layer
    X = MaxPooling2D(pool_size=(2,2), padding='valid', strides=(2,2), name = 'max2')(X)
    print("maxpool2: " + str(X.shape))

    #third layer
    X = Conv2D(128, kernel_size=(3,3), padding='same', strides=(1,1), name = 'conv31')(X)
    print("conv31:" +str(X.shape))
    X = Conv2D(128, kernel_size=(3,3), padding='same', strides=(1,1), name = 'conv32')(X)
    print("conv32:" +str(X.shape))
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2,2), padding='valid', strides=(2,2), name = 'max3')(X)
    print("maxpool3: " + str(X.shape)) 
#    
    
    #fourth layer
    X = Conv2D(128, kernel_size=(3,3), padding='same', strides=(1,1), name = 'conv41')(X)
    print("conv41:" +str(X.shape))
#    X = Conv2D(512, kernel_size=(3,3), padding='same', strides=(1,1), name = 'conv42')(X)
#    print("conv42:" +str(X.shape))
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2,2), padding='valid', strides=(2,2), name = 'max4')(X)
    print("maxpool4: " + str(X.shape))
#    
#    #fifth layer
#    X = Conv2D(128, kernel_size=(3,3), padding='same', strides=(1,1), name = 'conv51')(X)
#    print("conv51:" +str(X.shape))
##    X = Conv2D(512, kernel_size=(3,3), padding='same', strides=(1,1), name = 'conv52')(X)
##    print("conv52:" +str(X.shape))
#    X = BatchNormalization()(X)
#    X = Activation('relu')(X)
#    X = MaxPooling2D(pool_size=(2,2), padding='valid', strides=(2,2), name = 'max5')(X)
#    print("maxpool5: " + str(X.shape))
    
    #first fc layer
    X = Flatten()(X)
    print("fc1: " + str(X.shape))
    
#    #second fc layer
#    X = Dense(1024, activation = "relu", use_bias = True, kernel_initializer="glorot_normal")(X)
#    print("fc2: " + str(X.shape))
    
    #third fc layer
    X = Dense(512, activation = "relu", use_bias = True, kernel_initializer="glorot_uniform")(X)
    print("fc3: " + str(X.shape))
#    
    #fourth fc layer
    X = Dense(64, activation = "relu", use_bias = True, kernel_initializer="glorot_uniform")(X)
    print("fc4: " + str(X.shape))
    
    #fifth fc layer
    Y = Dense(4, activation = "softmax", kernel_initializer="glorot_uniform")(X)
    print("fc5: " + str(Y.shape))
    
    model = Model(input = X_input, outputs=Y, name = 'vgg16_model')
    
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

model = vgg16_model((224, 224 ,3))
model.compile(optimizer='rmsprop', \
              loss = 'categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x = train_X, y= train_Y, batch_size = 16, epochs = 30)
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

with open('hist_vgg16_baseline.pickle', 'wb') as handle:
    pickle.dump(history.history, handle)

with open('hist_vgg16_baseline.pickle', 'rb') as handle:
    b = pickle.load(handle)

print(b)


print(prediction)

