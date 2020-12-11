from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

embeddingfile='output/embeddings.pickle'
recognizerfile='output/recognizer.pickle'
labelencoderfile='output/le.pickle'

print("loading the embedded faces")
data=pickle.loads(open(embeddingfile,"rb").read())

print("encoding labels")
labelenc=LabelEncoder()
labels=labelenc.fit_transform(data["names"])

print("Training Model")
recog=SVC(C=1.0,kernel="linear",probability=True)
recog.fit(data["embeddings"],labels)

f=open(recognizerfile,"wb")
f.write(pickle.dumps(recog))
f.close()

f = open(labelencoderfile, "wb")
f.write(pickle.dumps(labelenc))
f.close()
