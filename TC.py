from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from time import time
import numpy as np

#Decision Trees Model
def DT(categories):
    start = time()
    # Load data
    twenty_train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)
    
    # First the data is vectorized
    # Then words are all arranged according to their ids frequencies with which they occur
    # Lastly the Decision Tree is added as a classifier.
    transforms = [('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', DecisionTreeClassifier())]

    # The pipeline is created using all the transforms and the classifier specified above.
    text_clf = Pipeline(transforms)
    # The classifier is trained over the training data
    _ = text_clf.fit(twenty_train.data, twenty_train.target)

    # The test data is extracted
    twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
    # The test data is passed to the model and predictions are made
    predicted = text_clf.predict(twenty_test.data)
    
    end = time()

    print("*Decision Tree Model*")
    print("Newsgroup Categories : ", categories)
    print("Accuracy : {}%".format(np.mean(predicted == twenty_test.target) * 100))
    print("Time Taken: %0.3fs" % (end - start))
    print("\n")


# Naive Bayes Model
def NBModel(categories):
    # Load data
    start = time()
    twenty_train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)

    # Pipeline (tokenizer => transformer => MultinomialNB classifier)
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
    text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

    # Evaluation on test set
    twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
    predicted = text_clf.predict(twenty_test.data)
    predicted = text_clf.predict(twenty_train.data)

    end=time()
    print("*Naive Bayes Model*")
    print("Newsgroup Categories : ", categories)
    print("Accuracy : {}%".format(np.mean(predicted == twenty_train.target) * 100))
    print("Time Taken: %0.3fs" % (end - start))
    print("\n")


#Linear SVM Model
def SVM(categories):
    start = time()

    # Load data
    twenty_train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)

    # Pipeline (tokenizer => transformer => linear SVM classifier)
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42))])
    _ = text_clf.fit(twenty_train.data, twenty_train.target)

    # Evaluation on test set
    twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
    predicted = text_clf.predict(twenty_test.data)
    end=time()
    print("*SVM Model*")
    print("Newsgroup Categories : ", categories)
    print("Accuracy : {}%".format(np.mean(predicted == twenty_test.target) * 100))
    print("Time Taken: %0.3fs" % (end - start))
    print("\n")
    

# Selected categories
categories = ["soc.religion.christian", "comp.graphics", "sci.med", "talk.politics.misc", "misc.forsale"]
    
DT(categories)
NBModel(categories)
SVM(categories)

# All categories
categories = ["comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware",
              "comp.windows.x", "rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey", "sci.crypt",
              "sci.electronics", "sci.med", "sci.space", "misc.forsale", "talk.politics.misc", "talk.politics.guns",
              "talk.politics.mideast", "talk.religion.misc", "alt.atheism", "soc.religion.christian"]

DT(categories)
NBModel(categories)
SVM(categories)
