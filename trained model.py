import os
from keras.preprocessing.image import ImageDataGenerator
import pickle

test_val_data = os.path.join('test_val')

test_val = len(os.listdir('test_val/test_photo'))

batch_size = 16
epochs = 100
IMG_HEIGHT = 150
IMG_WIDTH = 150

test_val_generator = ImageDataGenerator(rescale=1./255)

test_val_dir = os.path.join('test_val')


test_val_data_gen = test_val_generator.flow_from_directory(batch_size=1,
                                                              directory=test_val_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')


pickle_in = open("chest_xray_model", "rb")
classifier = pickle.load(pickle_in)

prediction = classifier.predict_generator(generator=test_val_data_gen, verbose=2, steps=1)

print(prediction)

