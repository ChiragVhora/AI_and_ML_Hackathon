'''
    this file contains functions for saving and retrieving neural network model , using jsom
'''


def cmn_load_nn_model(model_name):
    print(f"{model_name} model loading ...")
    from keras.models import model_from_json  # saving and loading model

    weight_file = f"{model_name}.h5"
    model_file = f'{model_name}.json'


    loaded_model_JSON = None
    with open(model_file, "r") as JSON_file:
        loaded_model_JSON = JSON_file.read()

    model = model_from_json(loaded_model_JSON)
    model.load_weights(weight_file)
    print("model loaded from disc successfully !")
    return model
    pass


def cmn_save_nn_model(model, model_name):
    print(f"{model_name} model saving...")
    model_JSON = model.to_json()

    weight_file = f"{model_name}.h5"
    model_file = f'{model_name}.json'

    with open(model_file, "w") as JSON_file:
        JSON_file.write(model_JSON)

    model.save_weights(weight_file)
    print("model saved successfully !")
    pass
