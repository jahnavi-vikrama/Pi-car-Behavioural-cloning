import numpy as np
from keras.models import load_model
from PIL import Image
import cv2
import os
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

model = load_model('model.h5')

img  = Image.open('IMG_8594.JPG')
plt.imshow(img)

def img_preprocess(img):
  img=img[1500:2500,0:4000,:]
  img=cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
  img=cv2.GaussianBlur(img,(3,3),0)
  img=cv2.resize(img,(200,66))
  img=img/255
  return img
#img = cv2.flip(img,1)
img = np.asarray(img)

image = img_preprocess(img)
plt.imshow(image)
image = np.array([image])

steering_angle = float(model.predict(image))
print(steering_angle)
