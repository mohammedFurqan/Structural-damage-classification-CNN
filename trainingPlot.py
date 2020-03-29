import pickle
import numpy as np
import matplotlib.pyplot as plt


with open('hist_alexnet_baseline.pickle', 'rb') as handle:
    hist_alexnet = pickle.load(handle)

with open('hist_vgg16_baseline.pickle', 'rb') as handle:
    hist_vgg16 = pickle.load(handle)
    
with open('hist_resnet50_baseline.pickle', 'rb') as handle:
    hist_resnet50 = pickle.load(handle)

with open('hist_vgg19_fine_tuning.pickle', 'rb') as handle:
    hist_vgg19_fine = pickle.load(handle)

with open('hist_vgg16_fine_tuning.pickle', 'rb') as handle:
    hist_vgg16_fine = pickle.load(handle)  

with open('hist_resnet50_fine_tuning.pickle', 'rb') as handle:
    hist_resnet50_fine = pickle.load(handle)

with open('hist_MobileNetV2_fine_tuning.pickle', 'rb') as handle:
    hist_MobileNetV2_fine = pickle.load(handle)

with open('hist_mobilenet_fine_tuning.pickle', 'rb') as handle:
    hist_mobilenet_fine = pickle.load(handle)

with open('hist_InceptionV3_fine_tuning.pickle', 'rb') as handle:
    hist_InceptionV3_fine = pickle.load(handle)

Fontsize = 40
Legend = 26
# loss function
loss_alex = hist_alexnet['loss']
loss_vgg16 = hist_vgg16['loss']
loss_resnet50 = hist_resnet50['loss']
loss_vgg19_fine  = hist_vgg19_fine['loss']
loss_vgg16_fine = hist_vgg16_fine['loss']
loss_resnet50_fine = hist_resnet50_fine['loss']
loss_MobileNetV2_fine = hist_MobileNetV2_fine['loss']
loss_mobilenet_fine = hist_mobilenet_fine['loss']
loss_InceptionV3_fine = hist_InceptionV3_fine['loss']
epo = np.arange(30) + 1;
               
plt.figure(figsize=(16,10))
plt.title('Training error',fontsize=Fontsize)
Tick = 26 
plt.plot(epo, loss_alex, label='Baseline-AlexNet',linewidth=4.0)
plt.plot(epo, loss_vgg16, label='Baseline-vgg16',linewidth=4.0)
plt.plot(epo, loss_resnet50, label='Baseline-ResNet50',linewidth=4.0)
plt.plot(epo, loss_vgg16_fine, label='VGG16',linewidth=4.0)
plt.plot(epo, loss_vgg19_fine, label='VGG19',linewidth=4.0)
plt.plot(epo, loss_resnet50_fine, label='ResNet50',linewidth=4.0)
plt.plot(epo, loss_mobilenet_fine, label='MobileNet',linewidth=4.0)
plt.plot(epo, loss_MobileNetV2_fine, label='MobileNetV2',linewidth=4.0)
plt.plot(epo, loss_InceptionV3_fine, label='InceptionV3',linewidth=4.0)
plt.legend(fontsize=Legend,ncol = 2, loc='upper right')
plt.xlabel('epoch', fontsize=Fontsize)
plt.ylabel('loss', fontsize=Fontsize)
plt.xticks(fontsize=Tick)
plt.yticks(fontsize=Tick)
plt.axis([0, 30, 0, 2.5])
plt.grid()
plt.show()

# training error

acc_alex = hist_alexnet['acc']
acc_vgg16 = hist_vgg16['acc']
acc_resnet50 = hist_resnet50['acc']
acc_vgg19_fine  = hist_vgg19_fine['acc']
acc_vgg16_fine = hist_vgg16_fine['acc']
acc_resnet50_fine = hist_resnet50_fine['acc']
acc_MobileNetV2_fine = hist_MobileNetV2_fine['acc']
acc_mobilenet_fine = hist_mobilenet_fine['acc']
acc_InceptionV3_fine = hist_InceptionV3_fine['acc']

plt.figure(figsize=(16,10))
plt.title('Training accuracy',fontsize=Fontsize)

plt.plot(epo, acc_alex, color='green', label='AlexNet',linewidth=4.0)
plt.plot(epo, acc_vgg16, color='red', label='vgg16',linewidth=4.0)
plt.plot(epo, acc_resnet50,  color='skyblue', label='ResNet50',linewidth=4.0)
plt.plot(epo, acc_vgg16_fine, label='VGG16',linewidth=4.0)
plt.plot(epo, acc_vgg19_fine, label='VGG19',linewidth=4.0)
plt.plot(epo, acc_resnet50_fine, label='ResNet50',linewidth=4.0)
plt.plot(epo, acc_mobilenet_fine, label='MobileNet',linewidth=4.0)
plt.plot(epo, acc_MobileNetV2_fine, label='MobileNetV2',linewidth=4.0)
plt.plot(epo, acc_InceptionV3_fine, label='InceptionV3',linewidth=4.0)
plt.legend(fontsize=Legend,ncol = 2, loc='lower right')
plt.xlabel('epoch', fontsize=Fontsize)
plt.ylabel('accuracy', fontsize=Fontsize)
plt.xticks(fontsize=Tick)
plt.yticks(fontsize=Tick)
plt.axis([0, 30, 0, 1])
plt.grid()
plt.show()




