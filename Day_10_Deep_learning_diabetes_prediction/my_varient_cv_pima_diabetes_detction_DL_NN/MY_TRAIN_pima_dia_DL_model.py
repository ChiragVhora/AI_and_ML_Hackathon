'''
    Logic
        1. initialize
        2. get best model by keras tuner :
            ref - https://youtu.be/Clo1HKB50Ug
                - https://www.youtube.com/watch?v=vvC15l4CY1Q
        3. save higher accuracy best model
'''
import time

from numpy import loadtxt                   # for loading csv dataset

from keras.models import Sequential         # for initializing blank NN
from keras.layers import Dense              # dense for making hidden and output layers
import keras_tuner as kt
'''
    installed 1.0.0 version then "pip install keras-tuner --upgrade" to remove error
    replace in code kerastuner to keras_tuner
     # more :  https://keras.io/keras_tuner/'''

###################### function ###################################

def save_pima_diabetes_NN_model(model):
    from keras.models import model_from_json  # saving and loading model
    model_JSON = model.to_json()
    with open("NN_pima_diabetes_model.json", "w") as JSON_file:
        JSON_file.write(model_JSON)
    model.save_weights("model.h5")
    print("model saved successfully !")
    pass

def save_test_data(test_data):
    with open("pima_diabetes_test_data.csv", "w") as test_csv:
        row_data = ""
        for row in test_data:
            i = 1
            for data in row:
                if i == 6 or i == 7:
                    row_data += f"{data},"
                else:
                    data = int(data)
                    row_data += f"{data},"
                i += 1

            # rows+= f'{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]},{row[6]},{row[7]},{row[8]}\n'
            row_data = row_data[:-1]    # removing extra ,
            row_data += "\n"

        test_csv.write(row_data)
    pass


def build_model(hp):
    tuner_first_layer_nodes = hp.Int("first layer nodes", min_value=8, max_value=20)

    input_layer_with_first_hidden_layer = Dense(tuner_first_layer_nodes, input_dim=8, activation='relu')

    for i in range(hp.Int("layers-2", min_value=1, max_value=3, step=1)):    # step is increment by
        tuner_hidden_layer_node = hp.Int("hidden_node_of_layer_"+str(i), min_value=8, max_value=16)
        tuner_hidden_L_activation_function = hp.Choice("hidden_act_of_layer_"+str(i), ['relu', 'sigmoid'])

        hidden_layer = Dense(tuner_hidden_layer_node, activation=tuner_hidden_L_activation_function)

    output_layer = Dense(1,
                         activation='sigmoid')  # activation sig-> bcs back propagation is not done at output ,i think

    NN_model = Sequential()  # init NN
    NN_model.add(input_layer_with_first_hidden_layer)
    NN_model.add(hidden_layer)
    NN_model.add(output_layer)

    # compile
    NN_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return NN_model
    pass
#########################################################
start = time.time()
print("loading dataset...")
csv_dataset = loadtxt("pima-indians-diabetes.csv", delimiter=',')   # loading CSV ( comma separated values )
X = csv_dataset[:, 0:8]
Y = csv_dataset[:, 8]

# last 20 rows for testing purpose
X_train_input_vals = X[: -20]
Y_train_output_vals = Y[: -20]

X_test_ip_val = X[-20:]
Y_test_op_val = Y[-20:]

test_data = csv_dataset[-20:]   # last 20 rows
save_test_data(test_data)
end = time.time()
time_taken = (end-start)
print(f"loading dataset finished in {time_taken} ms")


# print(f"input val : {X_train_input_vals}, output : {Y_train_output_vals}")
##################### keras tuner ####################################

def get_dir_path_to_save_models():
    import os
    path = os.path.normpath('C:/')
    return path
    pass


Dir_path = get_dir_path_to_save_models()    # normal name works fine but get error in kerastunner
                                            # err 1 : long path
                                            # err 2 : mixed slashes in path like : c/path\model\data
tuner = kt.RandomSearch(
    build_model,
    objective="val_accuracy",           # what is goal -> validation accuracy
    max_trials=40,                       # max random combinations - high no -> high computational overhead
    executions_per_trial=3,             # one combination this times - "same"
    directory=Dir_path,                 # for saving weights and model
    project_name="PD model"
)

# print("Search variables are : ", tuner.search_space_summary())
tuner.search(X_train_input_vals, Y_train_output_vals, epochs=5, validation_data=(X_test_ip_val, Y_test_op_val))
best_model = tuner.get_best_models()[0]

print("\n\n\n################ Results #################3")
print(tuner.results_summary())

# this will be performed here but little differently
# NN_model.fit(X_train_input_vals, Y_train_output_vals, epochs=epoch_OR_iteration, batch_size=batch_size)
# _, accuracy = NN_model.evaluate(X_train_input_vals, Y_train_output_vals)


#########################################################



save_pima_diabetes_NN_model(best_model)     # save model function for pima diabetes


#################### Dump - Ignore #####################################

# def build_model(hp):
#     # global X_train_input_vals
#     # global Y_train_output_vals
#     # loss_func = 'binary_crossentropy'
#     # optimizer = 'adam'
#     # activation_function = 'relu'  # relu basic idea: https://youtu.be/lJ4_tvwIVg8?t=369
#     # starting_input_nodes = 8
# 
#     # accuracy = 0
#     # flag = 0
#     # best_accuracy = 0
#     # best_model = None
#     # NN_model = None
#     # epoch_OR_iteration = 6
#     # batch_size = 10
#     # while accuracy < 85 or flag < 10:
#     tuner_first_layer_nodes = hp.Int("first layer nodes", min_value=8, max_value=20)
# 
#     input_layer_with_first_hidden_layer = Dense(tuner_first_layer_nodes, input_dim=8, activation='relu')
# 
#     for i in range(hp.Int("layers", min_value=1, max_value=4, step=1)):    # step is increment by
#         tuner_hidden_layer_node = hp.Int("hidden_node_of_layer_"+str(i), min_value=8, max_value=16)
#         tuner_hidden_L_activation_function = hp.Choice("hidden_act_of_layer_"+str(i), ['relu', 'sigmoid'])
# 
#         hidden_layer = Dense(tuner_hidden_layer_node, activation=tuner_hidden_L_activation_function)
# 
#     output_layer = Dense(1,
#                          activation='sigmoid')  # activation sig-> bcs back propagation is not done at output ,i think
# 
#     NN_model = Sequential()  # init NN
#     NN_model.add(input_layer_with_first_hidden_layer)
#     NN_model.add(hidden_layer)
#     NN_model.add(output_layer)
# 
#     # compile
#     NN_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     # NN_model.fit(X_train_input_vals, Y_train_output_vals, epochs=epoch_OR_iteration, batch_size=batch_size)
#     # _, accuracy = NN_model.evaluate(X_train_input_vals, Y_train_output_vals)
# 
#     # if best_accuracy < accuracy:
#     #     best_accuracy = accuracy
#     #     best_model = NN_model
#     # print(f"Accuracy of NN model : {accuracy * 100}")
#     #
#     return NN_model
#     pass
#########################################################