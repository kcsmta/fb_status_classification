import io
import os
import time

import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer

from pyvi import ViTokenizer

stopwords_encoding = "utf-8"

stopwords_path = "./stopwords/stopwords-vi.txt"
# stopwords_path = sys.argv[1]

path_to_dict = "dict_bow.pkl"
# path_to_dict = sys.argv[2]

path_to_model = "SVMmodel.sav"
# path_to_model = sys.argv[3]

path_to_check_folder = "./test_fb_data"
# path_to_check_folder = sys.argv[4]

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

def remove_stop_word(text, stopwords_path):
    with io.open(stopwords_path, "r", encoding = stopwords_encoding) as f:
        stopwords = set([w.strip().replace(' ', '_') for w in f.readlines()])
    for w in stopwords:
        word = " " + w + " "
        word = word.lower()
        if text.find(word) != -1:
            new_text = text.replace(word, " ")
            text = new_text
    return new_text

corpus = []
names = []

for file in os.listdir(path_to_check_folder):
    path_to_file = os.path.join(path_to_check_folder, file)
    with io.open(path_to_file,'r',encoding='utf-8') as f:
        text = f.read()
    corpus.append(text)
    names.append(file)

def extract_BoW(corpus, path_to_dict):
    # print("Loading dictionary...")
    vocab = pickle.load(open(path_to_dict, "rb"))
    vectorizer = CountVectorizer(decode_error="replace", vocabulary=vocab)
    # print ("Loading dictionary complete!!!")
    X = []
    for i in range(len(corpus)):
        text = corpus[i]
        doc = [text]
        X_data = vectorizer.transform(doc).toarray()
        X.append(X_data)

    # print ("Extracting BoW from data finished!!!")
    return np.array(X, dtype='int32')

# Extract Bag of Word features
X = extract_BoW(corpus, path_to_dict)

nsamples_val, nx, ny = X.shape
X = X.reshape((nsamples_val,nx*ny))

model = pickle.load(open(path_to_model, "rb"))

pred = model.predict(X)

time.sleep(15)
for i, x in enumerate(names):
    print("File {} belongs to class {}".format(names[i], pred[i]))