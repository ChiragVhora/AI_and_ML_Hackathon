import os

from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Dense, Flatten, Dropout
from keras.utils import np_utils
   # class to matrix

from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# variables #######################################
from common_python_files.get_image_path import cmn_get_all_image_path_from_folder

list_of_img_np = []
labels = []
classes = {43}
# cwd =
train_folder = r""

# Training data obtaining #######################################
print("Obtaining images respective labels")
for root, Dir, files in os.walk(train_folder):
    folder_name = os.path.basename(root)
    images_paths = cmn_get_all_image_path_from_folder(train_folder, True)
    for image_path in images_paths:
        image = Image.open(image_path)
        res_img = image.resize((30, 30))
        np_img = np.array(res_img)

        list_of_img_np.append(np_img)
        labels.append(folder_name)

        print(f"{os.path.basename(image_path)} loaded from folder {folder_name}")

print("dataset loaded.")

# converting list into np array
np_image_data = np.array(list_of_img_np)
np_label = np.array(labels)
assert np_image_data.shape == np_label.shape , "dimension not same so prob in upper part, in append"

# splitting train snd test data
X_train, X_test, Y_train, Y_test = train_test_split(np_image_data, np_label, test_size=0.2, random_state=30)

print(X_train.shape, X_test.shape)
print(Y_train.shape, Y_test.shape)

# converting labels into encoding
categorical_y_train = np_utils.to_categorical(Y_train, len(classes))
categorical_y_test = np_utils.to_categorical(Y_test, len(classes))

# model making ######################
road_sign_model = Sequential()
road_sign_model.add(Conv2D(filters=32, kernel_size=(5, 5), activation="relu", input_shape=(30, 30, 3)))
# input in rgb
road_sign_model.add(Conv2D(filters=32, kernel_size=(5, 5), activation="relu",))
road_sign_model.add(AveragePooling2D(pool_size=(2, 2)))
road_sign_model.add(Dropout(0.25))

road_sign_model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu",))
road_sign_model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu",))
road_sign_model.add(AveragePooling2D(pool_size=(2, 2)))
road_sign_model.add(Dropout(0.25))

road_sign_model.add(Flatten())

road_sign_model.add(Dense(256, activation="relu"))

road_sign_model.add(Dropout(rate=0.5))

road_sign_model.add(Dense(units=len(classes), activation="softmax"))

print("[info] model initialized .")
# compiling model ##########################
road_sign_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["Accuracy"])

# training model #######################
result_history = road_sign_model.fit(
    X_train,
    Y_train,
    batch_size=32,
    epochs=5,
    validation_data=(X_test, Y_test),
    validation_steps=20
)

RD_model_name = "Road_sign_recognition_model.h5"
road_sign_model.save(RD_model_name)

# graphing the result and saving it ####################
# for accuracy
plt.figure(0)
plt.plot(result_history.history["accuracy"], label="Training Accuracy")
plt.plot(result_history.history["val_accuracy"], label="Val Accuracy")
plt.title("Accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.savefig("Accuracy.png")

# for loss
plt.figure(0)
plt.plot(result_history.history["loss"], label="Training Loss")
plt.plot(result_history.history["val_loss"], label="Val Loss")
plt.title("Loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.savefig("Loss.png")

# done !
print("process completed !")

