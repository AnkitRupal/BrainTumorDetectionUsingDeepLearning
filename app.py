import streamlit as st
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub 
import streamlit as st
import tensorflow_hub as hub
import cv2
from PIL import Image, ImageOps


st. set_page_config(layout="wide")


@st.cache(allow_output_mutation=True)
def load_model():
    modelPath = './my_model.hdf5'
    model=tf.keras.models.load_model(modelPath ,custom_objects={'KerasLayer':hub.KerasLayer})
    return model



def import_and_predict(image_data, model):
    size = (224,224)    
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    test_image=image/255
    test_image = test_image[np.newaxis,...]
    result = model.predict(test_image)
    return result


st.write("""
         # Brain Tumor Detection
         """
         )
file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)

model = load_model()

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    str = ("{:.2f} chances of Brain Tumor".format( 100 * np.max(predictions)))
    st.text(str)
