import io
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from pyvi import ViTokenizer

from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score

seed = 7
np.random.seed(seed)

MAX_SENT_LENGTH = 100
MAX_SENTS = 15
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2
N_CLASSES = 2
content_encoding = "utf-8"
stopwords_encoding = "utf-8"
dictionary_encoding = "utf-8"

data_train = pd.read_csv('./data_fb_status.csv', escapechar='\\', encoding =content_encoding)
print data_train.shape

def clear_text(text):
    text = text.lower()
    text = text.replace("\n", ". ")
    text = text.replace("..",".")
    text = text.replace("...",".")
    text = text.replace(". .", ".")
    text = text.replace("?",".")
    text = text.replace("!",".")
    text = text.replace("-"," ")
    text = text.replace(","," ")
    text = text.replace("/"," ")
    text = text.replace("("," ")
    text = text.replace(")"," ")
    text = text.replace(":"," ")
    text = " ".join(text.split())
    return text

def word_segmentation(text):
    text = ViTokenizer.tokenize(text)
    if not text.startswith(" "):
        text = " " + text
    return text

def remove_stop_word(text):
    new_text = text
    stopwords_path = "./stopwords/stopwords-vi.txt"
    with io.open(stopwords_path, "r", encoding = stopwords_encoding) as f:
        stopwords = set([w.strip().replace(' ', '_') for w in f.readlines()])
    for w in stopwords:
        word = " " + w + " "
        word = word.lower()
        if text.find(word) != -1:
            new_text = text.replace(word, " ")
            text = new_text
    return new_text

labels = []
corpus = []

for idx in range(data_train.content.shape[0]):
    text = data_train.content[idx]
    text = clear_text(text)
    text = word_segmentation(text)
    text = remove_stop_word(text)
    corpus.append(text)

    labels.append(data_train.ID[idx])

def make_dictionary(corpus, save_to):
    print ("Making dictionary...")
    vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=MAX_NB_WORDS)
    vectorizer.fit(corpus)
    pickle.dump(vectorizer.vocabulary_,open(save_to,"wb"))
    print ("Making dictionary complete!!!")

# step 1: Create Dictionary
make_dictionary(corpus, "dict_bow.pkl")

def extract_BoW(corpus, labels, path_to_dict):
    print("Loading dictionary...")
    vectorizer = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open(path_to_dict, "rb")))
    print ("Loading dictionary complete!!!")
    X = []
    y = []
    for i in range(len(corpus)):
        text = corpus[i]
        doc = [text]
        X_data = vectorizer.transform(doc).toarray()
        y_data = labels[i]
        X.append(X_data)
        y.append(y_data)

    print ("Extracting BoW from data finished!!!")
    return np.array(X, dtype='int32'), np.array(y, dtype='int32')

# step 2: Extract Bag of Word features
X, y = extract_BoW(corpus, labels, "dict_bow.pkl")

indices = np.arange(len(corpus))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]
nb_validation_samples = int(VALIDATION_SPLIT * len(corpus))
nb_test_samples = int(TEST_SPLIT *len(corpus)) + nb_validation_samples

x_val = X[:nb_validation_samples]
x_test = X[nb_validation_samples:nb_test_samples]
x_train = X[nb_test_samples:]
y_val = y[:nb_validation_samples]
y_test = y[nb_validation_samples:nb_test_samples]
y_train = y[nb_test_samples:]

nsamples_test, nx, ny = x_test.shape
x_test = x_test.reshape((nsamples_test,nx*ny))

nsamples_train, nx, ny = x_train.shape
x_train = x_train.reshape((nsamples_train,nx*ny))

nsamples_val, nx, ny = x_val.shape
x_val = x_val.reshape((nsamples_val,nx*ny))

print ("Training SVM model")
model = LinearSVC()
# model = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0)
# model = MultinomialNB()
# model = LogisticRegression(random_state=0)

model.fit(x_train, y_train)
results = model.predict(x_val)
acc = 100*accuracy_score(y_val, results)
print "Validation Accuracy = {:.3}%".format(acc)

results = model.predict(x_test)
acc = 100*accuracy_score(y_test, results)
print "Test Accuracy = {:.3}%".format(acc)

y_pred = model.predict(x_test)
test_f1 = f1_score(y_test, y_pred, average=None)
test_precision = precision_score(y_test, y_pred, average=None)
test_recall = recall_score(y_test, y_pred, average=None)
print (" \ntest_f1: {} \ntest_precision: {} \ntest_recall: {}".format(test_f1, test_precision, test_recall))

# save model
pickle.dump(model, open("SVMmodel.sav", "wb"))
