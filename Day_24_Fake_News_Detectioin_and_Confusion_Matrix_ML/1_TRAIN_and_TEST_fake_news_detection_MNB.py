from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import itertools

# var#################################
FAKE = "FAKE"
REAL = "REAL"

# function ##############################################
def plot_confusion_matrix(cm, classes, normalize=False,
                          title="Confusion matrix", cmap=plt.cm.get_cmap("Blues")):
    # for plotting
    plt.imshow(cm, interpolation="nearest", cmap=cmap)   # cmap = color
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))    # Real, Fake (2x2)
    plt.xticks(tick_marks, classes, rotation=45)    # rotation to Real,Fake
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print("Confusion Matrix, Without Normalization")

    thresh = cm.max() / 2.      # for color thresh , white text in blue part and black text in white part
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()
    pass

# generate pandas dataframe ###################################3
fake_news = "Fake.csv"
true_news = "True.csv"
use_col = ["title", "text"]
# for true
true_dataframe = pd.read_csv(true_news,usecols=use_col)
true_dataframe["Label"] = [REAL for i in enumerate(true_dataframe["title"])]
# print(true_dataframe.head())
# print(true_dataframe.shape)

# for false
false_dataframe = pd.read_csv(fake_news,usecols=use_col)
false_dataframe["Label"] = [FAKE for i in enumerate(false_dataframe["title"])]
# print(false_dataframe.head())
# print(false_dataframe.shape)

# merged dataframe
dataframe = pd.concat([false_dataframe, true_dataframe])
# print(dataframe.head())
# print(dataframe.shape)


# splitting into train test ######################################
X = dataframe.drop("Label", axis=1)
# print(X.head())
Y = dataframe.Label
# print(Y.head())

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1000)
X_train, X_test, Y_train, Y_test = train_test_split(X["text"], Y, test_size=0.3, random_state=1)

# print(X_train.head())
# print(X_test.head())
# err
print(type(X_test))
X_test = pd.concat([ X["text"][:100], X["text"][-100:] ])       # first hun. and last hund.
Y_test = pd.concat([ Y[:100], Y[-100:] ])       # first hun. and last hund.
# print(type(X_test))
# X_test = X["text"][-100:]  #X["text"][:100]      # first hun. and last hund.
# Y_test = Y[-100:]       #Y[:100]     # first hun. and last hund.
print(X_test)
print(Y_test)

# extracting features ############################################3
count_vectorizer = CountVectorizer(stop_words="english")    # vectorizer for simple english words created,
                                                            # for removing worda like in,the,an, ,....
X_vectorized_train_data = count_vectorizer.fit_transform(X_train)

X_vectorized_test_data = count_vectorizer.transform(X_test)

print("Vec names len : ", len(count_vectorizer.get_feature_names()))

# print("     Vectorzed Train  : ", X_vectorized_train_data)

# vectorized datadrame
# vectorized_datafraame = pd.DataFrame(X_vectorized_train_data.A, columns=count_vectorizer.get_feature_names())
# print(vectorized_datafraame.head())

# Model training #################################################
FN_model = MultinomialNB()

FN_model.fit(X_vectorized_train_data, Y_train)


# testing ########################################################
predict = FN_model.predict(X_vectorized_test_data)

# results
score = metrics.accuracy_score(Y_test, predict)

print("accuracy : %0.3f" % score)

# draw confusion metrix ###############################
confusion_metrix = metrics.confusion_matrix(Y_test, predict, labels=[FAKE, REAL])
plot_confusion_matrix(confusion_metrix, classes=[FAKE, REAL])



