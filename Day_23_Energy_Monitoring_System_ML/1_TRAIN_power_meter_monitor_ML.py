from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split    # splitting data
from sklearn.model_selection import cross_val_score   # for evaluating
from sklearn.model_selection import StratifiedKFold

# variables ##############################################
data_file_path = "power_monitor_data.csv"
col_names = ["Current", "Voltage", "Power", "Output Condition"]
dataset = read_csv(data_file_path, names=col_names,skiprows=1)

# data splitting ##################################
array =dataset.values
X = array[ :, 0:3]
y = array[ :, 3]

X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.30, random_state=3)

# models making ##################################
# all_models = {}

# all_models["LR"] = LogisticRegression(solver="liblinear", multi_class="ovr")
# all_models["LR"] = LinearDiscriminantAnalysis()
# all_models["KNN"] = KNeighborsClassifier()
CART_model = DecisionTreeClassifier()
# all_models["GNB"] = GaussianNB()
# all_models["SVM"] = SVC(gamma="auto")

CART_model.fit(X_train, Y_train)

# saving the model #####################################
import pickle
pickle_model_path = "CART_DT_power_monitor.pkl"
with open(pickle_model_path, "wb") as f:
    pickle.dump(CART_model, f)

# result ################################################
result = CART_model.score(X_val, Y_val)
print(result)

# Testing ##########################################333
value = [[211.23, 0.3425, 151]]
predictions = CART_model.predict(value)
print(predictions)

