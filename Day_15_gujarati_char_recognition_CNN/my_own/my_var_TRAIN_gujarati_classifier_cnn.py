from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential     # model init

from keras.layers import Conv2D, MaxPool2D  # for 2D con
from keras.layers import MaxPooling2D           # for reduce dimension
from keras.layers import BatchNormalization     # for normalizing the input
from keras.layers import Dense                  # for hidden ANN
from keras.layers import Flatten                # for feature map to arr
from keras.layers import Dropout                # remove dead neuron

from keras.preprocessing.image import ImageDataGenerator    # data gen

# variables #############################################
from common_python_files.save_OR_load_model_into_jason import cmn_save_nn_model

model_name = "my_var_Gujarati_classifier_cnn_model"

classes = [ "ALA", "ANA", "B", "BHA", "CH", "CHH", "D", "DA", "DH", "DHA", "F", "G", "GH", "GNA", "H", "J", "JH", "K", "KH", "KSH", "L", "M", "N", "P", "R", "S", "SH", "SHH", "T", "TA", "TH", "THA", "V", "Y"]

def train_gujarati_classifier_cnn():
    global classes
    global model_name
    # Layers ##########################################################

    kernel, activation_fn, max_pool_size = ((3, 3), "relu", (2, 2))

    cnn = Sequential()
    kernelSize = (3, 3)
    ip_activation = 'relu'
    ip_conv_0 = Conv2D(filters=32, kernel_size=kernelSize, input_shape=(128, 128, 1), activation=ip_activation)
    cnn.add(ip_conv_0)
    # Add the next Convolutional+Activation layer
    ip_conv_0_1 = Conv2D(filters=64, kernel_size=kernelSize, activation=ip_activation)
    cnn.add(ip_conv_0_1)

    # Add the Pooling layer
    pool_0 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")
    cnn.add(pool_0)
    ip_conv_1 = Conv2D(filters=64, kernel_size=kernelSize, activation=ip_activation)
    cnn.add(ip_conv_1)
    ip_conv_1_1 = Conv2D(filters=64, kernel_size=kernelSize, activation=ip_activation)
    cnn.add(ip_conv_1_1)

    pool_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")
    cnn.add(pool_1)
    # Let's deactivate around 20% of neurons randomly for training
    drop_layer_0 = Dropout(0.2)
    cnn.add(drop_layer_0)
    flat_layer_0 = Flatten()
    cnn.add(Flatten())
    # Now add the Dense layers
    h_dense_0 = Dense(units=128, activation=ip_activation, kernel_initializer='uniform')
    cnn.add(h_dense_0)
    # Let's add one more before proceeding to the output layer
    h_dense_1 = Dense(units=64, activation=ip_activation, kernel_initializer='uniform')
    cnn.add(h_dense_1)
    op_activation = 'softmax'
    output_layer = Dense(units=len(classes), activation=op_activation, kernel_initializer='uniform')
    cnn.add(output_layer)

    opt = 'adam'
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    # Compile the classifier using the configuration we want
    cnn.compile(optimizer=opt, loss=loss, metrics=metrics)
    print(cnn.summary())

    # compile the model ##########################################################
    # from tensorflow import keras
    # optimiser = keras.optimizers.Adam(learning_rate=0.01,clipnorm=1.0)
    # guj_char_classifier_cnn_model = cnn
    # # loss = keras.losses.SparseCategoricalCrossentropy()
    # # loss = keras.losses.categoricalCrossentropy()
    # guj_char_classifier_cnn_model.compile(optimizer=optimiser, loss="categorical_crossentropy", metrics=["accuracy"])

    # data generator ###########################################################33
    train_datagen = ImageDataGenerator(
        rescale=None,
        shear_range=0.2,
         zoom_range=0.2,
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
        classes=classes,
        color_mode="grayscale"
    )

    val_set_batch = val_datagen.flow_from_directory(
        val_Dir,
        target_size=(128, 128),
        batch_size=128,
        classes=classes,
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
        EarlyStopping(monitor='accuracy', patience=100),
        ModelCheckpoint(filepath=model_name, monitor="accuracy", save_best_only=True)
    ]
    cnn.fit_generator(
                                 training_set_batch,
                                 steps_per_epoch=18,
                                 epochs=6,
                                 validation_data=val_set_batch,
                                 validation_steps=50,
                                 callbacks=callback
    )

    # save model ##########################################################################3
    cmn_save_nn_model(cnn, model_name)
    pass

train_gujarati_classifier_cnn()