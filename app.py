import streamlit as st
import keras
import cv2 as cv
import numpy as np
import os


#WORk_DIR = '/workspaces/icecream-streamlit-combo/work'
WORk_DIR = "work"
LABELS   = ['Ben & Jerry\'s','Breyer\'s','Häagen-Dazs','Talenti']
IMG_SIZE = 32


def get_image_label(img_local_full_path):
 
    # read image from local to server binary (array format)
    img_arr = cv.imread(img_local_full_path)[...,::-1] #convert BGR to RGB format

    # image prep for model input
    IMG_HEIGHT = IMG_SIZE
    IMG_WIDTH  = IMG_SIZE

    resized_img_arr = cv.resize(img_arr, (IMG_HEIGHT, IMG_WIDTH)) # Reshaping images to preferred size
    
    flatten_image = resized_img_arr.reshape(-1, IMG_HEIGHT*IMG_WIDTH*3)
    
    # reshape/flatten input array for model input
    reshape_dim = (-1, 32, 32, 3)
    flatten_image = flatten_image.reshape(reshape_dim)
    
    # pre trained model to use
    modelname = '/workspaces/icecream-streamlit-combo/modelz/icecream/'
    
    # load the model
    trained_model = keras.models.load_model(modelname)
    
    # predict image label index
    pred_dist = trained_model.predict(flatten_image)
    pred_index = np.argmax(pred_dist[0])
    
    # predicted label 
    label = LABELS[pred_index]
    
    return label


#
#
#  Capture and Process the input
#
#

st.title("Iscream or NotCream")
st.markdown("## By: [Aadi Patangi](https://github.com/AadiPatangi)")
image_file = st.file_uploader("Upload an image: ",type=["png","jpg"])

if image_file:
    #st.markdown('Upload complete!')
    # Hide filename on UI
    st.markdown('''
        <style>
            .uploadedFile {display: none}
        <style>''',
        unsafe_allow_html=True)


image_full_path = None
#
#Write input image to local server dir (WORK_DIR)
if image_file is not None:
    #
    image_full_path = os.path.join(WORk_DIR,image_file.name)
    with open(image_full_path,"wb") as f:
        f.write(image_file.getbuffer())

# Process file stored to WORK_DIR
if image_full_path is not None:
    
    pred_label = get_image_label(image_full_path)
    
    if os.path.exists(image_full_path):
        os.remove(image_full_path)
    #
    st.header("Uploaded: "+pred_label)
    st.image(image_file,caption=None)
    