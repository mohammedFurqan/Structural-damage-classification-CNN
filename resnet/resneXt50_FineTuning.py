import numpy as np
import tensorflow as tf
import keras
from keras import regularizers
from keras_applications.resnext import ResNeXt50
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, Dense, Activation, ZeroPadding2D,\
BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout,\
GlobalMaxPool2D, GlobalAveragePooling2D
from keras import backend as K
import pickle
import matplotlib.pyplot as plt
import matplotlib


def load_dataset():
    X_set = np.load("phi2018task7/X_train.npy") # (2632, 224, 224, 3)
    
    N = X_set.shape[0]
    Y_set = np.load("phi2018task7/Y_train.npy") # (2632,)

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



# begining fine tuning

# create the base pre-trained model
input_tensor = Input(shape=(224, 224, 3))
base_model = ResNeXt50(input_tensor=input_tensor, include_top=False, weights='imagenet')

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer

# -----------------------------------------
x = Dropout(0.1)(x)
x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
x = Dropout(0.1)(x)
x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
# -----------------------------------------
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(4, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

# train the model on the new data for a few epochs
model.fit(x = train_X, y= train_Y, batch_size = 8, epochs = 2)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:173]:
   layer.trainable = False
for layer in model.layers[173:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
history = model.fit(x = train_X, y= train_Y, batch_size = 8, epochs = 30)

prediction = model.evaluate(x = test_X, y = test_Y, batch_size=8)

print(history)

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

with open('hist_resneXt50_fine_tuning.pickle', 'wb') as handle:
    pickle.dump(history.history, handle)

with open('hist_resneXt50_fine_tuning.pickle', 'rb') as handle:
    b = pickle.load(handle)

print(b)

print(prediction)


#------------------------------------------------------------
for i in range(17):
    namepath = 'resneXt50_output_0/' + str(i) + '.png'
    matplotlib.image.imsave(namepath, test_X[i])
#--------------------------------------------------------------
#

# print intermediate feature
layer_name1 = 'conv1_pad'
intermediate_layer_model1 = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name1).output)
intermediate_output1 = intermediate_layer_model1.predict(test_X[0:17,:,:,:])
print(intermediate_output1.shape)

n = intermediate_output1.shape[0]
for i in range(n):
    m = intermediate_output1.shape[3]
    for j in range(m):
        namepath = 'resneXt50_output_1/' + str(i) + '_'+ str(j) + '.png'
        matplotlib.image.imsave(namepath, intermediate_output1[i,:,:,j])

#-----------------------------------------------------
layer_name2 = 'conv1'
intermediate_layer_model2 = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name2).output)
intermediate_output2 = intermediate_layer_model2.predict(test_X[0:17,:,:,:])
print(intermediate_output2.shape)

for i in range(n):
    m = intermediate_output2.shape[3]
    for j in range(m):
        namepath = 'resneXt50_output_2/' + str(i) + '_'+ str(j) + '.png'
        matplotlib.image.imsave(namepath, intermediate_output2[i,:,:,j])

#----------------------------------
layer_name3 = 'bn_conv1'
intermediate_layer_model3 = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name3).output)
intermediate_output3 = intermediate_layer_model3.predict(test_X[0:17,:,:,:])
print(intermediate_output3.shape)

for i in range(n):
    m = intermediate_output3.shape[3]
    for j in range(m):
        namepath = 'resneXt50_output_3/' + str(i) + '_'+ str(j) + '.png'
        matplotlib.image.imsave(namepath, intermediate_output3[i,:,:,j])


layer_name4 = 'activation_1'
intermediate_layer_model4 = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name4).output)
intermediate_output4 = intermediate_layer_model4.predict(test_X[0:17,:,:,:])
print(intermediate_output4.shape)

for i in range(n):
    m = intermediate_output4.shape[3]
    for j in range(m):
        namepath = 'resneXt50_output_4/' + str(i) + '_'+ str(j) + '.png'
        matplotlib.image.imsave(namepath, intermediate_output4[i,:,:,j])
