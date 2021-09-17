from numpy import loadtxt                   # for loading csv dataset

from keras.models import Sequential         # for initializing blank NN
from keras.layers import Dense              # dense for making hidden and output layers
###################### function ###################################

def save_pima_diabetes_NN_model(model):
    model_JSON = model.to_json()
    with open("NN_pima_diabetes_model.json", "w") as JSON_file:
        JSON_file.write(model_JSON)
    model.save_weights("model.h5")
    print("model saved successfully !")
    pass
#########################################################

csv_dataset = loadtxt("pima-indians-diabetes.csv", delimiter=',')   # loading CSV ( comma separated values )
X_input_vals = csv_dataset[:, 0:8]
Y_output_vals = csv_dataset[:, 8]

# print(f"input val : {X_input_vals}, output : {Y_output_vals}")
#########################################################

nodes = 12
starting_input_nodes = 8
activation_function = 'relu'        # relu basic idea: https://youtu.be/lJ4_tvwIVg8?t=369
loss_func = 'binary_crossentropy'
optimizer = 'adam'
epoch_OR_iteration = 6
batch_size = 10

Input_layer_with_first_hidden_layer = Dense(nodes, input_dim=starting_input_nodes, activation=activation_function)
nodes = 8
hidden_layer = Dense(nodes,activation=activation_function)
nodes = 1   # binary classification at output
output_layer = Dense(nodes, activation='sigmoid')       # activation sig-> bcs back propagation is not done at output ,i think

NN_model = Sequential()     # init NN
NN_model.add(Input_layer_with_first_hidden_layer)
NN_model.add(hidden_layer)
NN_model.add(output_layer)
'''
 I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.'''
#########################################################

# compile and training
NN_model.compile(loss=loss_func, optimizer=optimizer, metrics=['accuracy'])
NN_model.fit(X_input_vals, Y_output_vals, epochs=epoch_OR_iteration, batch_size=batch_size)

_,accuracy = NN_model.evaluate(X_input_vals, Y_output_vals)
print(f"Accuracy of NN model : {accuracy*100}")

save_pima_diabetes_NN_model(NN_model)     # save model function for pima diabetes
