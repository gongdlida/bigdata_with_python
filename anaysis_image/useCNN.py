# CNN을 이용한 숫자 이미지 분류
import numpy as np
from mnist import MNIST
from tensorflow import keras
mndata = MNIST("anaysis_image/test.png")
images, labels = mndata.load_training()

trainImages = mndata.train_images()
trainLabels = mndata.train_labels()

print(trainImages.shape) # (60000, 28, 28) 
print(trainLabels.shape) # (60000,)

testImages = mndata.test_images()
testLabels = mndata.test_labels()

print(testImages.shape) #(10000, 28, 28)
print(testLabels.shape) #(10000,)

# Normalize the images.
trainImages = (trainImages / 255) - 0.5
testImages = (testImages / 255) - 0.5

# Reshaping the images with 1 channel
trainImages = trainImages.reshape(60000,28,28,1) 
testImages = testImages.reshape(10000,28,28,1)