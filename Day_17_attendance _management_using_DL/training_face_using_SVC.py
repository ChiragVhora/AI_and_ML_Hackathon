from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

# input and o/p for training SVC : embedding, names
embedding_file = "output/embeddings.pickle"

print("Loading embeddings ...")
data = pickle.load(open(embedding_file, "rb").read())

print("Encoding labels ...")    # names as labels
LabelEnc = LabelEncoder()
labels = LabelEnc.fit_transform(data["names"])

print("Training model ...")
recognizer = SVC(C=1., kernel="linear", probability=True)
recognizer.fit(data["encodings"], labels)

# saving recognizer and label encoder file
recognizer_file = "output/recognizer.pickle"    # contain trained model
with open(recognizer_file, "wb") as f:
    f.write(pickle.dumps(recognizer))

LabelEnc_file = "output/label_encoder.pickle"      # contain encoded names
with open(LabelEnc_file, "wb") as f:
    f.write(pickle.dumps(LabelEnc))