from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential     # model init

from keras.layers import Conv2D                 # for 2D con
from keras.layers import MaxPooling2D           # for reduce dimension
from keras.layers import BatchNormalization     # for normalizing the input
from keras.layers import Dense                  # for hidden ANN
from keras.layers import Flatten                # for feature map to arr
from keras.layers import Dropout                # remove dead neuron

from keras.preprocessing.image import ImageDataGenerator    # data gen

# variables #############################################
from common_python_files.save_OR_load_model_into_jason import cmn_save_nn_model

model_name = "Gujarati_classifier_cnn_model"

classes = [ "ALA", "ANA", "B", "BHA", "CH", "CHH", "D", "DA", "DH", "DHA", "F", "G", "GH", "GNA", "H", "J", "JH", "K", "KH", "KSH", "L", "M", "N", "P", "R", "S", "SH", "SHH", "T", "TA", "TH", "THA", "V", "Y"]

def train_gujarati_classifier_cnn():
    global classes
    global model_name
    # Layers ##########################################################

    kernel, activation_fn, max_pool_size = ((3, 3), "relu", (2, 2))

    convolution_layer_1 = Conv2D(32, kernel_size=kernel, activation=activation_fn, input_shape=(128, 128, 1))
    max_pool_layer_2 = MaxPooling2D(pool_size=max_pool_size)
    batch_norm_layer_3 = BatchNormalization()

    convolution_layer_4 = Conv2D(64, kernel_size=kernel, activation=activation_fn, input_shape=(128, 128, 3))
    max_pool_layer_5 = MaxPooling2D(pool_size=max_pool_size)
    batch_norm_layer_6 = BatchNormalization()

    convolution_layer_7 = Conv2D(64, kernel_size=kernel, activation=activation_fn, input_shape=(128, 128, 3))
    max_pool_layer_8 = MaxPooling2D(pool_size=max_pool_size)
    batch_norm_layer_9 = BatchNormalization()

    convolution_layer_10 = Conv2D(96, kernel_size=kernel, activation=activation_fn, input_shape=(128, 128, 3))
    max_pool_layer_11 = MaxPooling2D(pool_size=max_pool_size)
    batch_norm_layer_12 = BatchNormalization()

    convolution_layer_13 = Conv2D(32, kernel_size=kernel, activation=activation_fn, input_shape=(128, 128, 3))
    max_pool_layer_14 = MaxPooling2D(pool_size=max_pool_size)
    batch_norm_layer_15 = BatchNormalization()

    dropout_layer_16 = Dropout(0.2)

    flatten_layer_17 = Flatten()

    hidden_layer_18 = Dense(128, activation=activation_fn)

    dropout_layer_19 = Dropout(0.3)

    hidden_layer_20 = Dense(45, activation=activation_fn)

    output_layer_21 = Dense(len(classes), activation=activation_fn)

    # adding layer to model ###########################################
    guj_char_classifier_cnn_model = Sequential()
    guj_char_classifier_cnn_model.add(convolution_layer_1)
    guj_char_classifier_cnn_model.add(max_pool_layer_2)
    guj_char_classifier_cnn_model.add(batch_norm_layer_3)

    guj_char_classifier_cnn_model.add(convolution_layer_4)
    guj_char_classifier_cnn_model.add(max_pool_layer_5)
    guj_char_classifier_cnn_model.add(batch_norm_layer_6)

    guj_char_classifier_cnn_model.add(convolution_layer_7)
    guj_char_classifier_cnn_model.add(max_pool_layer_8)
    guj_char_classifier_cnn_model.add(batch_norm_layer_9)

    guj_char_classifier_cnn_model.add(convolution_layer_10)
    guj_char_classifier_cnn_model.add(max_pool_layer_11)
    guj_char_classifier_cnn_model.add(batch_norm_layer_12)

    guj_char_classifier_cnn_model.add(convolution_layer_13)
    guj_char_classifier_cnn_model.add(max_pool_layer_14)
    guj_char_classifier_cnn_model.add(batch_norm_layer_15)

    guj_char_classifier_cnn_model.add(dropout_layer_16)

    guj_char_classifier_cnn_model.add(flatten_layer_17)

    guj_char_classifier_cnn_model.add(hidden_layer_18)

    guj_char_classifier_cnn_model.add(dropout_layer_19)

    guj_char_classifier_cnn_model.add(hidden_layer_20)

    guj_char_classifier_cnn_model.add(output_layer_21)

    # compile the model ##########################################################
    from tensorflow import keras
    optimiser = keras.optimizers.Adam(learning_rate=0.01,clipnorm=1.0)
    # loss = keras.losses.SparseCategoricalCrossentropy()
    # loss = keras.losses.categoricalCrossentropy()
    guj_char_classifier_cnn_model.compile(optimizer=optimiser, loss="categorical_crossentropy", metrics=["accuracy"])

    # data generator ###########################################################33
    train_datagen = ImageDataGenerator(
         rescale=None,
         shear_range= 0.2,
         zoom_range= 0.2,
         horizontal_flip= True
     )

    val_datagen = ImageDataGenerator(rescale=1./255)

    # batch ###########################################################33

    train_Dir  = r"C:\Users\chira\PycharmProjects\AI_and_ML_Hackathon\Day_15_gujarati_char_recognition_CNN\Dataset\train"
    val_Dir = r"C:\Users\chira\PycharmProjects\AI_and_ML_Hackathon\Day_15_gujarati_char_recognition_CNN\Dataset\validation"

    training_set_batch = train_datagen.flow_from_directory(
        train_Dir,
        target_size=(128, 128),
        batch_size=128,
        class_mode="categorical",
        color_mode="grayscale"
    )

    val_set_batch = val_datagen.flow_from_directory(
        val_Dir,
        target_size=(128, 128),
        batch_size=128,
        class_mode="categorical",
        color_mode="grayscale"
    )

    # label_1 = training_set_batch.labels
    # label_2 = val_set_batch.labels
    # print(label_1)
    # print(label_2)
    # fit model ######################################################################333
    callback = [
        # EarlyStopping(monitor='val_loss', patience=50),  # if val_loss is not being efficient till the patience reached
        # then terminate training process and save best model till now
        # ModelCheckpoint(filepath=model_name, monitor='val_loss', save_best_only=True),
        EarlyStopping(monitor='val_accuracy', patience=100),
        ModelCheckpoint(filepath=model_name, monitor="val_accuracy", save_best_only=True)
    ]
    guj_char_classifier_cnn_model.fit_generator(
                                 training_set_batch,
                                 steps_per_epoch=18,
                                 epochs=10,
                                 validation_data=val_set_batch,
                                 validation_steps=50,
                                 callbacks=callback
    )

    # save model ##########################################################################3
    cmn_save_nn_model(guj_char_classifier_cnn_model, model_name)
    pass

train_gujarati_classifier_cnn()