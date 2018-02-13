from sklearn.feature_extraction.text import CountVectorizer
import nltk
import random
import nltk.classify.util
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC
import pickle
import numpy as np
from sklearn.model_selection import KFold

documents = []
words = []

useless_words = stopwords.words("english")
useless_words1 = [',','.','-','_','(',')',':','?',';','"']

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append([movie_reviews.words(fileid),category])
    
#documents[0]
#len(documents)

save_documents = open("documents1.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()
print("Document saved")
random.shuffle(documents)

for word in movie_reviews.words():
    if word not in useless_words and word not in useless_words1:
        words.append(word.lower())
print(len(words))

words_Freq=nltk.FreqDist(words)

#word_features = list(words_Freq.keys())[:8000]
word_features = list(words_Freq.keys())[:5000]
save_word_features = open("word_features1.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()
print("word features saved")

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

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", nltk.classify.accuracy(classifier, testing_set))

print(classifier.show_most_informative_features(10))

kf = KFold(n_splits=10)
sum = 0
for train, test in kf.split(featuresets):
    train_data = np.array(featuresets)[train]
    test_data = np.array(featuresets)[test]
    Kfoldclassifier = nltk.NaiveBayesClassifier.train(train_data)
    sum += nltk.classify.accuracy(classifier, test_data)
average = sum/10
print("K-fold cross validation accuracy:",average)


MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB classifier accuracy:", nltk.classify.accuracy(MNB_classifier, testing_set))

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
nltk.classify.accuracy(SVC_classifier,testing_set)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
nltk.classify.accuracy(LinearSVC_classifier,testing_set)

try_rev = '''Amazing piece of art. Worldclass. Awesome. Worth it.A bit shady.'''
try_word = word_tokenize(try_rev)

def create_word_features(words):
    useful_words = [word for word in words if word not in useless_words and word not in useless_words1]
    my_dictionary = dict([(word, True) for word in useful_words])
    return my_dictionary
    
	
try_word = create_word_features(try_word)
classifier.classify(try_word)

save_classifier = open("naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

save_classifier = open("MNB_classifier.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

save_classifier = open("SVC_classifier.pickle","wb")
pickle.dump(SVC_classifier, save_classifier)
save_classifier.close()

save_classifier = open("LinearSVC_classifier.pickle","wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

save_classifier = open("Kfoldclassifier.pickle","wb")
pickle.dump(Kfoldclassifier, save_classifier)
save_classifier.close()

neg = 0
pos = 0
sentence = "Awesome movie. I like it. It is so bad."
sentence = sentence.lower()
sentences = sentence.split('.')   # these are actually list of sentences

for sent in sentences:
    if sent != "":
        words = [word for word in sent.split(" ")]
        classResult = classifier.classify(create_word_features(words))
        if classResult == 'neg':
            neg = neg + 1
        if classResult == 'pos':
            pos = pos + 1
        print(str(sent) + ' --> ' + str(classResult))
		
short_pos = open("unlabeled.txt", encoding="utf8").read()
for sent in short_pos.split('\n'):
    if sent != "":
        words = [word.lower() for word in sent.split(" ")]
        classResult = classifier.classify(create_word_features(words))
        if classResult == 'neg':
            neg = neg + 1
        if classResult == 'pos':
            pos = pos + 1
        print(str(sent) + ","+str(classResult))
        save_documents = open("saveddata2.txt","a")
        save_documents.write(str(sent) + ","+str(classResult))
        save_documents.close()
print("pos:neg  " + str(pos)+ ":" + str(neg))
pos_res = pos/(pos+neg)
print ("total positive reviews:",pos_res)



