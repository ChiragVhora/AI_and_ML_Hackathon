from keras.models import Sequential  # for creating blank NN obj
from keras.layers import Conv2D  # for 1st Con. layer
from keras.layers import MaxPool2D  # for 2nd max_polling layer
from keras.layers import Flatten  # for 3rd laer for converting to flatten arr
from keras.layers import Dense  # for 4rd dense layer and 5th o/p layer
from keras.preprocessing.image import ImageDataGenerator  # for generation of new data from out dataset


# functions ################################


def save_CNN_model(model):
    from keras.models import model_from_json  # saving and loading model
    model_JSON = model.to_json()
    with open("CNN_model.json", "w") as JSON_file:
        JSON_file.write(model_JSON)
    model.save_weights("cnn_model.h5")
    print("model saved successfully !")
    pass


# ############################################
# creating layers
convolution_layer = Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu')
max_polling_layer = MaxPool2D(pool_size=(2, 2))
flattener_layer = Flatten()
hidden_layer = Dense(128, activation='relu')
output_layer = Dense(1, activation='sigmoid')  # for binary o/p

# creating model and adding layers
cnn_model = Sequential()
cnn_model.add(convolution_layer)
cnn_model.add(max_polling_layer)
cnn_model.add(flattener_layer)
cnn_model.add(hidden_layer)
cnn_model.add(output_layer)

# compiling the model
cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# data generation ( training , validation )
train_data_gen = ImageDataGenerator(rescale=1. / 255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

validation_data_gen = ImageDataGenerator(rescale=1. / 255)

# data set
train_set = train_data_gen.flow_from_directory('Dataset/train',
                                               target_size=(64, 64),
                                               batch_size=8,
                                               class_mode='binary')  # for binary classification

validation_set = train_data_gen.flow_from_directory('Dataset/validation',
                                                    target_size=(64, 64),
                                                    batch_size=8,
                                                    class_mode='binary')  # for binary classification

# fit the model
cnn_model.fit_generator(train_set,
                        steps_per_epoch=10,
                        epochs=5,
                        validation_data=validation_set)

# save model
from common_python_files.save_OR_load_model_into_jason import cmn_save_nn_model
model_name = "cnn_nn_joker_and_thanos_recognition"
cmn_save_nn_model(cnn_model, model_name)