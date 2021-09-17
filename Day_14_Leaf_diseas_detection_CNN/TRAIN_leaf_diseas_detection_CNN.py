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
model_name = "leaf_disease_cnn_model"
classes = [ "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy", "Blueberry___healthy", "Cherry_(including_sour)___healthy", "Cherry_(including_sour)___Powdery_mildew", "Corn_(maize)___Common_rust_", "Corn_(maize)___healthy", "Corn_(maize)___Northern_Leaf_Blight", "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___healthy", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___healthy", "Potato___Late_blight", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew", "Strawberry___healthy", "Strawberry___Leaf_scorch", "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___healthy", "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"]

# Layers ##########################################################

kernel, activation_fn, max_pool_size = ((3, 3), "relu", (2, 2))

convolution_layer_1 = Conv2D(32, kernel_size=kernel, activation=activation_fn, input_shape=(128, 128, 3))
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

dropout_layer_19 = Dropout(0.2)

hidden_layer_19_5 = Dense(76, activation=activation_fn)


output_layer_20 = Dense(len(classes), activation=activation_fn)

# adding layer to model ###########################################
leaf_cnn_model = Sequential()
leaf_cnn_model.add(convolution_layer_1)
leaf_cnn_model.add(max_pool_layer_2)
leaf_cnn_model.add(batch_norm_layer_3)

leaf_cnn_model.add(convolution_layer_4)
# leaf_cnn_model.add(max_pool_layer_5)
leaf_cnn_model.add(batch_norm_layer_6)

leaf_cnn_model.add(convolution_layer_7)
leaf_cnn_model.add(max_pool_layer_8)
leaf_cnn_model.add(batch_norm_layer_9)

leaf_cnn_model.add(convolution_layer_10)
leaf_cnn_model.add(max_pool_layer_11)

leaf_cnn_model.add(batch_norm_layer_12)

leaf_cnn_model.add(convolution_layer_13)
leaf_cnn_model.add(max_pool_layer_14)
leaf_cnn_model.add(batch_norm_layer_15)

leaf_cnn_model.add(dropout_layer_19)

leaf_cnn_model.add(flatten_layer_17)


leaf_cnn_model.add(hidden_layer_18)

leaf_cnn_model.add(dropout_layer_19)

leaf_cnn_model.add(hidden_layer_19_5)

leaf_cnn_model.add(output_layer_20)

# compile the model ##########################################################
leaf_cnn_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print(leaf_cnn_model.summary())

# data generator ###########################################################33
train_datagen = ImageDataGenerator(
     rescale=None,
     shear_range= 0.2,
     zoom_range= 0.2,
     horizontal_flip= True
 )

test_datagen = ImageDataGenerator(rescale=1./255)

# batch ###########################################################33

train_Dir  = r"C:\Users\chira\Desktop\ML_ai_hackathon_aug_21\Day_14_leaf_diesese_detection\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"
val_Dir = r"C:\Users\chira\Desktop\ML_ai_hackathon_aug_21\Day_14_leaf_diesese_detection\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid"
training_set_batch = train_datagen.flow_from_directory(
    train_Dir,
    target_size=(128, 128),
    batch_size=100,
    class_mode="categorical",
    classes=classes
)

val_set_batch = train_datagen.flow_from_directory(
    val_Dir,
    target_size=(128, 128),
    batch_size=64,
    class_mode="categorical",
    classes=classes
)

# label_1 = training_set_batch.labels
# label_2 = val_set_batch.labels
# print(label_1)
# print(label_2)
callback = [
        # EarlyStopping(monitor='val_loss', patience=50),  # if val_loss is not being efficient till the patience reached
        # then terminate training process and save best model till now
        # ModelCheckpoint(filepath=model_name, monitor='val_loss', save_best_only=True),
        EarlyStopping(monitor='accuracy', patience=35),
        EarlyStopping(monitor='loss', patience=35),
        ModelCheckpoint(filepath=model_name, monitor="accuracy", save_best_only=True)
    ]
# fit model ######################################################################333
leaf_cnn_model.fit_generator(
                             training_set_batch,
                             steps_per_epoch=10,
                             epochs=50,
                             validation_data=val_set_batch,
                             validation_steps=10)

# save model ##########################################################################3
cmn_save_nn_model(leaf_cnn_model, model_name)
