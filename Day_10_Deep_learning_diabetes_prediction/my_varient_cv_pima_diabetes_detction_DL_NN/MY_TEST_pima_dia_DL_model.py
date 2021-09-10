import time

from numpy import loadtxt  # for loading csv dataset

from keras.models import Sequential         # for initializing blank NN
from keras.layers import Dense              # dense for making hidden and output layers
###################### function ###################################

def load_pima_diabetes_NN_model():
    from keras.models import model_from_json  # saving and loading model

    loaded_model_JSON = None
    with open("NN_pima_diabetes_model.json", "r") as JSON_file:
        loaded_model_JSON = JSON_file.read(loaded_model_JSON)

    model = model_from_json(loaded_model_JSON)
    model.load_weights("model.h5")
    print("model loaded from disc successfully !")
    return model
    pass
#########################################################

print("Loading test data....")
start_t = time.time()

csv_dataset = loadtxt("pima_diabetes_test_data.csv", delimiter=',')   # loading CSV ( comma separated values )
X_input_vals = csv_dataset[:, 0:8]
Y_output_vals = csv_dataset[:, 8]

end_t = time.time()
time_taken = end_t-start_t
print(f"test data loaded successfully in {time_taken} ms")

# print(f"input val : {X_input_vals}, output : {Y_output_vals}")
#########################################################

# model loading
NN_model = load_pima_diabetes_NN_model()

# predicting
# predictions = NN_model.predict_classes(X_input_vals)      not working for me because i have higher version
start_t = time.time()

predictions = NN_model.predict(X_input_vals)    # this method gives list of tuples as [(P_of_false, P_of_True), ...]

end_t = time.time()
time_taken = end_t-start_t
# print(all_predictions)
# classes_x=np.argmax(predict_x, axis=1)

success_count = 0
total = len(predictions)
for i in range(0, total):
    prediction_val = round(predictions[i][0])      # rounding val for checking
    if prediction_val == Y_output_vals[i]:
        success_count += 1
    print(f"For data : {X_input_vals[i]} prediction : {predictions[i][0]}, Accuracy (pred. vs orig.) :{prediction_val} vs {Y_output_vals[i]} ")

print("\n\n###################### result ###############################")
print(f"\nsuccess ration : {success_count}/{total} , {success_count/total*100} % \nprocessed in {time_taken} ms")