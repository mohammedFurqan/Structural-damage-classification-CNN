import tensorflow as tf
import pickle
import keras
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPool2D, GlobalAveragePooling2D
from keras.models import Model
import keras.backend as K
K.set_image_data_format("channels_last")

import numpy as np
import matplotlib.pyplot as plt
import pickle

#plt.imshow(X_train[0])
#plt.show()

def load_dataset():
    X_set = np.load("../data_phi2018/task7_X_train.npy") # (2632, 224, 224, 3)
    N = X_set.shape[0]
    Y_set = np.load("../data_phi2018/task7_y_train.npy") # (2632,)
    
    # shuffle the data 
    arr = np.arange(N)
    np.random.shuffle(arr)
    
    train_x_orig = X_set[arr[0:2400]]
    train_y_orig = (Y_set[arr[0:2400]][:,0]).astype(int)
    test_x_orig = X_set[arr[2400: -1]]
    test_y_orig = (Y_set[arr[2400: -1]][:,0]).astype(int)
    train_y_orig = train_y_orig.reshape((1, train_y_orig.shape[0]))
    test_y_orig = test_y_orig.reshape((1, test_y_orig.shape[0]))
    
    classes = np.arange(np.max(test_y_orig)+1)
    print("classes range: " + str(np.min(test_y_orig)) + " " + str(np.max(test_y_orig)))
    return train_x_orig, train_y_orig, test_x_orig, test_y_orig, classes

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def alexnet_model(input_shape):
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
    X = ZeroPadding2D(padding=(2, 2), data_format=None)(X_input)
    print("zeropadding:" +str(X.shape))
    
    # First layer
    # X = Dropout(0.1)(X)
    X = Conv2D(96, kernel_size=(11,11), padding='valid', strides=(4,4), name = 'conv1')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    print("conv1:" +str(X.shape))
    
    # First layer followed by a max pooling layer
    X = MaxPooling2D(pool_size=(3,3), padding='valid', strides=(2,2), name = 'max1')(X)
    print("maxpool1: " + str(X.shape))
    
    # Second layer, 256
    X = Conv2D(256, kernel_size=(5,5), padding='same', strides=(1,1), name = 'conv2')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)    
    print("conv2:" +str(X.shape))
    
    # Second layer followed by a max pooling layer
    X = MaxPooling2D(pool_size=(3, 3), padding='valid', strides=(2,2), name = 'max2')(X)
    print("maxpool2: " + str(X.shape))
    
    # Third layer, 384
    X = Conv2D(384, kernel_size=(3,3), padding='same', strides=(1,1), name = 'conv3')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    print("conv3:" +str(X.shape))
        
    # Fourth layer, 384
    X = Conv2D(384, kernel_size=(3,3), padding='same', strides=(1,1), name = 'conv4')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    print("conv4:" +str(X.shape))
    
    # Fifth layer, 256
    X = Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), name = 'conv5')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    print("conv5:" +str(X.shape))   
    
    # Fifth layer followed by a max pooling layer
    X = MaxPooling2D(pool_size=(3, 3), padding='valid', strides=(2,2), name = 'max3')(X)
    print("maxpool3: " + str(X.shape))
    
    # First fc layer
    X = Flatten()(X)
    print("fc1: " + str(X.shape))
    #    #second fc layer
    #    X = Dense(4096, activation = "relu", use_bias = True, kernel_initializer="glorot_normal")(X)
    #    print("fc2: " + str(X.shape))
    
    # Third fc layer
    X = Dense(512, activation = "relu", use_bias = True, kernel_initializer="glorot_uniform")(X)
    print("fc3: " + str(X.shape))
    
    # Fourth fc layer
    X = Dense(64, activation = "relu", use_bias = True, kernel_initializer="glorot_uniform")(X)
    print("fc4: " + str(X.shape))
    
    # Fifth fc layer
    Y = Dense(4, activation = "softmax", kernel_initializer="glorot_uniform")(X)
    print("fc5: " + str(Y.shape))
    
    model = Model(input = X_input, outputs=Y, name = 'alexnet_model')
    
    return model


train_x_orig, train_y_orig, test_x_orig, test_y_orig, classes = load_dataset()
# Normalize image vectors
train_X = train_x_orig/255.
test_X = test_x_orig/255.

# Reshape
train_Y = convert_to_one_hot(train_y_orig, 4).T
test_Y = convert_to_one_hot(test_y_orig, 4).T

print("number of class " + str(np.max(train_y_orig)))
print("number of class " + str(np.max(train_y_orig)))

print("train_x_orig " + str(train_X.shape))
print("test_x_orig " + str(test_X.shape))
print("train_y_orig " + str(train_y_orig.shape))
print("test_y_orig " + str(train_y_orig.shape))
print("train_y " + str(train_Y.shape))
print("test_y " + str(test_Y.shape))
    
print("classes " + str(classes))

model = alexnet_model((224, 224 ,3))
model.compile(optimizer='sgd', loss = 'categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x = train_X, y= train_Y, epochs = 30)
prediction = model.evaluate(x = test_X, y = test_Y)

plt.plot(history.history['acc']) 
plt.title('model accuracy w/o dropout')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

plt.plot(history.history['loss']) 
plt.title('model loss w/o dropout')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

# with open('/saved_training_model', 'wb') as out_file:
#     pickle.dump(history.history, out_file)

with open('alexnet_baseline.pickle', 'wb') as handle:
    pickle.dump(history.history, handle)

with open('alexnet_baseline.pickle', 'rb') as handle:
    b = pickle.load(handle)

print(b)
print(prediction)



