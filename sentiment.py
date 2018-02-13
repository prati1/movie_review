import nltk
import random
#from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


documents = []
all_words = []
useless_words = stopwords.words("english")
useless_words1 = [',','.','-','_','(',')',':','?',';','"']
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append([movie_reviews.words(fileid),category])




word_features5k_f = open("pickle/word_features1.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()

def create_word_features(words):
    useful_words = [word for word in words if word not in useless_words and word not in useless_words1]
    my_dictionary = dict([(word, True) for word in useful_words])
    return my_dictionary

featuresets = []
for (review,category) in documents:
    words = set([word for word in review if word not in useless_words and word not in useless_words1])
    features = {}
  
    for w in word_features:
        features[w] = w in words
    featuresets.append([features,category])
len(featuresets)

training_set = featuresets[:1400]
testing_set = featuresets[1400:]



open_file = open("pickle/naivebayes.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()


open_file = open("pickle/MNB_classifier.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()



open_file = open("pickle/MNB_classifier.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()


open_file = open("pickle/LinearSVC_classifier.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()


open_file = open("pickle/Kfoldclassifier.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()


voted_classifier = VoteClassifier(
                                  classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)




#def sentiment(text):
#    words = set([word for word in text if word not in useless_words and word not in useless_words1])
#    features = {}
  
#    for w in word_features:
#        features[w] = w in words
#    return voted_classifier.classify(features),voted_classifier.confidence(features)
	
#print(sentiment("Best movie ever"))

def sentiment(text):
	short_pos = open(text, encoding="utf8").read()
	pos = 0
	neg = 0
	for sent in short_pos.split('\n'):
		if sent != "":
			words = [word.lower() for word in sent.split(" ")]
			classResult = voted_classifier.classify(create_word_features(words))
			if classResult == 'neg':
				neg = neg + 1
			if classResult == 'pos':
				pos = pos + 1
			print(str(sent) + ","+str(classResult))
			save_documents = open("data/output/result.txt","a")
			save_documents.write(str(sent) + ","+str(classResult) + '\n')
			save_documents.close()
	print("pos:neg  " + str(pos)+ ":" + str(neg))
	pos_res = (pos/(pos+neg))*100
	print ("Total positive reviews:",pos_res,"%")
	
text = input("Input the file to be analyzed, place in in the same file as sentiment.py: ")
sentiment(text)
