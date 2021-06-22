from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageOps
#import cv2
import streamlit as st
import tensorflow as tf

st.set_option('deprecation.showfileUploaderEncoding', False)

# Loaded models are saved to cache, to not have to load every time


@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('64_D_16.h5')
    return model


model = load_model()

st.write("""
    # Batman Classification
    """)

file = st.file_uploader("Please upload an image", type=["jpg", "png"])


def import_and_predict(image_data, model):
    size = (224, 224)
    #img = image.load_img(image_data, target_size=size)
    img = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    #img = image.img_to_array(img)
    img = np.asarray(img)
    img = preprocess_input(img)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names = ["Not Batman", "Batman"]
    result = class_names[np.where(predictions > 0.5, 1, 0)[0][0]]
    string = "The image is: " + result + ' ' + "({:.2f}% confidence)".format(predictions[0][0])
    
    st.success(string)
