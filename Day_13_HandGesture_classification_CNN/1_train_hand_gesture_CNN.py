from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten

from keras.preprocessing.image import ImageDataGenerator

# making layers ########
# input will be 256x256
# neurons = (150, 100, 64, 32, 32) # 0.25 accuracy
neurons = (32, 63, 128, 256, 150) #  loss: 1.7830 - accuracy: 0.2125 - val_loss: 1.7646 - val_accuracy: 0.2857
# neurons = (150, 100, 64, 32, 16) # loss: 5.1523 - accuracy: 0.1719
# neurons = (200, 100, 64, 32, 32) # loss: 5.2891 - accuracy: 0.1354
# neurons = (128, 100, 64, 32, 32) # loss: 4.3762 - accuracy: 0.1725 - val_loss: 4.9355 - val_accuracy: 0.2321
# neurons = (256, 100, 64, 32, 32) # loss: 5.9490 - accuracy: 0.1914
# neurons = (150, 100, 40, 32, 32)    # for 200 : loss: nan - accuracy: 0.1335
# neurons = (256, 50, 32, 32, 24)    # for 200 : loss: nan - accuracy: 0.1335
# neurons = (256, 42, 32, 32, 12)    # for 200 : loss: nan - accuracy: 0.1335
(kernel, input_shape, activation_fn) = ( (3, 3), (256, 256, 1), "relu")

neuron_count = neurons[0]
convolution_layer_1_1 = Conv2D(neuron_count, kernel_size=kernel, input_shape=input_shape, activation=activation_fn)
max_pooling_layer_1_2 = MaxPooling2D(pool_size=(2, 2))

neuron_count = neurons[1]
convolution_layer_2_3 = Conv2D(neuron_count, kernel_size=kernel, activation=activation_fn)
convolution_layer_3_4 = Conv2D(neuron_count, kernel_size=kernel, activation=activation_fn)
max_pooling_layer_2_5 = MaxPooling2D(pool_size=(2, 2))

neuron_count = neurons[2]
convolution_layer_4_6 = Conv2D(neuron_count, kernel_size=kernel, activation=activation_fn)
max_pooling_layer_3_7 = MaxPooling2D(pool_size=(2, 2))

neuron_count = neurons[3]
convolution_layer_5_8 = Conv2D(neuron_count, kernel_size=kernel, activation=activation_fn)
max_pooling_layer_4_9 = MaxPooling2D(pool_size=(2, 2))

flatten_layer_10 = Flatten()

neuron_count = neurons[4]
hidden_layer_11 = Dense(units=neuron_count, activation=activation_fn)    # 150

hidden_layer_2_12 = Dense(units=100, activation=activation_fn)    # 150
# hidden_layer_3_13 = Dense(units=50, activation=activation_fn)    # 150
# hidden_layer_3_14 = Dense(units=25, activation=activation_fn)    # 150

dropout_layer_12 = Dropout(0.25)

output_layer_13 = Dense(units=6, activation=activation_fn)


# ################### add layer to model ##########################
cnn_model = Sequential()

cnn_model.add(convolution_layer_1_1)
cnn_model.add(max_pooling_layer_1_2)

cnn_model.add(convolution_layer_2_3)
cnn_model.add(convolution_layer_3_4)
cnn_model.add(max_pooling_layer_2_5)
#
cnn_model.add(convolution_layer_4_6)
cnn_model.add(max_pooling_layer_3_7)

cnn_model.add(convolution_layer_5_8)
cnn_model.add(max_pooling_layer_4_9)

cnn_model.add(flatten_layer_10)
cnn_model.add(hidden_layer_11)

cnn_model.add(hidden_layer_2_12)
# cnn_model.add(hidden_layer_3_13)
# cnn_model.add(hidden_layer_3_14)

cnn_model.add(dropout_layer_12)

cnn_model.add(output_layer_13)

# ################# Compile model #################
cnn_model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])

# ################# Data Generation object with setting #################
train_data_gen_obj = ImageDataGenerator(rescale=1. / 255,
                                        rotation_range=12.,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        zoom_range=0.15,
                                        horizontal_flip=True)

validation_data_gen_obj = ImageDataGenerator(rescale=1. / 255)

# ################# Data set using above object #################
# classes = ['01_palm', '02_l', '03_fist', '04_fist_moved', "05_thumb", "06_index", "07_ok", "08_palm_moved", "09_c", "10_down"]
classes = ["index finger", "ok gesture", "palm", "pinky finger", "thumb down", "thumb up"]
train_f_path = r"C:\Users\chira\PycharmProjects\AI_and_ML_Hackathon\Day_13_HandGesture_classification\Handgesture_Dataset\train"
validation_f_path = r"C:\Users\chira\PycharmProjects\AI_and_ML_Hackathon\Day_13_HandGesture_classification\Handgesture_Dataset\validation"

target_size = (input_shape[0], input_shape[1])
training_dataset = train_data_gen_obj.flow_from_directory(directory=train_f_path,
                                                          target_size=target_size,
                                                          color_mode='grayscale',
                                                          batch_size=8,
                                                          classes=classes,
                                                          class_mode='categorical'
                                                          )

validation_dataset = validation_data_gen_obj.flow_from_directory(directory=validation_f_path,
                                                                 target_size=target_size,
                                                                 color_mode='grayscale',
                                                                 batch_size=8,
                                                                 classes=classes,
                                                                 class_mode='categorical'
                                                                 )

# ################# Callback list - early stopping and check point #################
model_name = 'hand_gesture_cnn_model.h5'
callback = [
    EarlyStopping(monitor='val_loss', patience=15),  # if val_loss is not being efficient till the patience reached
    # then terminate training process and save best model till now
    # ModelCheckpoint(filepath=model_name, monitor='val_loss', save_best_only=True),
    EarlyStopping(monitor='val_accuracy', patience=15),
    ModelCheckpoint(filepath=model_name, monitor="val_accuracy", save_best_only=True)
]

# ##############3## checkpoints ################################
# filepath=r"Check_points\weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]

# ################# fitting model from generator data #################
cnn_model.fit_generator(training_dataset,
                        steps_per_epoch=90,
                        epochs=10,
                        validation_data=validation_dataset,
                        validation_steps=7,
                        callbacks=callback)

'''
    for dataset : https://www.gti.ssr.upm.es/data/MultiModalHandGesture_dataset
        i used : https://www.kaggle.com/gti-upm/leapgestrecog
'''
