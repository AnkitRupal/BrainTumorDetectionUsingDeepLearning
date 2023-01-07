# IMPORTING LIBRARIES
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub 




# LOADING TEST DATASET
path = './archive/'
datagen = ImageDataGenerator(rescale=1./255,validation_split=0.0)
testing_set = datagen.flow_from_directory(directory=path,
                                            target_size=(224,224),
                                            color_mode="rgb",
                                            subset="training",
                                            class_mode="binary",
                                            batch_size=32,
                                            shuffle=True)

# IMPORTING MODEL
modelPath = './my_model.hdf5'
model=tf.keras.models.load_model(modelPath ,custom_objects={'KerasLayer':hub.KerasLayer})

# MODEL EVALUATION
loss = model.evaluate(testing_set)
print(f"Accuracy of the model on testing data : {loss[1]*100}%\nLoss of the model on testing data : {loss[0]}")