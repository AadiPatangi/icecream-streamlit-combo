import streamlit as st
from PIL import Image
import keras
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

st.title("Iscream or NotCream")
st.header("By: Aadi Patangi")
image = st.file_uploader("Upload an image: ",type=["png","jpg"])
image = Image.open(image)
st.image(image, caption='Uploaded File')
modelname = '/workspaces/icecream-streamlit-combo/modelz/icecream'
#model.save(modelname,save_format='h5')

trained_model = keras.models.load_model(modelname)
IMG_SIZE = 32
LABELS = ['bj','breyers','hd','talenti']
IMG_HEIGHT = IMG_SIZE
IMG_WIDTH  = IMG_SIZE

imgx = '/workspaces/icecream-streamlit-combo/projdata/testss/breyers/10_breyers.png'

img_arr = cv.imread(imgx)[...,::-1] #convert BGR to RGB format
resized_arr = cv.resize(img_arr, (IMG_HEIGHT, IMG_WIDTH)) # Reshaping images to preferred size
img_to_predict = resized_arr

flatten_image = img_to_predict.reshape(-1, IMG_HEIGHT*IMG_WIDTH*3)

res = batch_input_shape=(-1, 32, 32, 3)

flatten_image = flatten_image.reshape(res)
pred_dist = trained_model.predict(flatten_image)

pred_index = np.argmax(pred_dist[0])
pred_label = LABELS[pred_index]
data = np.asarray(img_arr)

imgplot = plt.imshow(data)
plt.title(pred_label)
st.image(image,caption=pred_label)
