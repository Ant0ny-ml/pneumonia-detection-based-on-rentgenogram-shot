
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import os
import pickle


train_data = os.path.join('train')
test_data = os.path.join('test')
validation_data = os.path.join('val')


total_train_pneumo = len(os.listdir('train/PNEUMONIA'))
total_train_normal = len(os.listdir('train/NORMAL'))
total_test_pneumo = len(os.listdir('test/PNEUMONIA'))
total_test_normal = len(os.listdir('test/NORMAL'))
valid_pneumo = len(os.listdir('val/PNEUMONIA'))
valid_normal = len(os.listdir('val/NORMAL'))




batch_size = 16
epochs = 100
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
test_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_dir = os.path.join('train')
test_dir = os.path.join('test')
val_dir = os.path.join('val')


total_train = total_train_pneumo + total_train_normal
total_test = total_test_pneumo + total_test_normal
total_validation = valid_pneumo + valid_normal


train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=test_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

validation_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=val_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (IMG_HEIGHT, IMG_WIDTH, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


optimizer = Adam(lr = 0.0001)
early = EarlyStopping(monitor="val_loss", mode="min", patience=3)
callbacks_list = [early]
classifier.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer=optimizer)
history = classifier.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=validation_data_gen,
    validation_steps=total_test // batch_size,
    callbacks=callbacks_list
)
optimizer = Adam(lr = 0.0001)
early = EarlyStopping(monitor="val_loss", mode="min", patience=3)
callbacks_list = [early]
classifier.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer=optimizer)
history = classifier.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=validation_data_gen,
    validation_steps=total_test // batch_size,
    callbacks=callbacks_list
)

with open("chest_xray_model", "wb") as f:
    pickle.dump(classifier, f)
