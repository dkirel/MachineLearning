import nltk
import pickle

from nltk.classify import ClassifierI
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.metrics.scores import precision, recall
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from statistics import mode


class VoteClassifier(ClassifierI):

    def __init__(self, *classifier_classes):
        self.classifier_classes = classifier_classes
        self.classifiers = []

    def classify(self, features):
        if self.classifiers:
            votes = [c.classify(features) for c in self.classifiers]
            return mode(votes)
        else:
            return 'Classifier has not been trained or loaded'

    def classify_text(self, text):
        features = self.find_features(text)
        return self.classify(features), self.confidence(features)

    def confidence(self, features):
        votes = [c.classify(features) for c in self.classifiers]
        return votes.count(mode(votes))/len(votes)

    def load_pickle(self):
        # Word Features
        open_file = open('pickled_files/word_features.pickle', 'rb')
        self.word_features = pickle.load(open_file)
        open_file.close()
        
        # Normal Naive Bayes
        open_file = open('pickled_files/NaiveBayesClassifier.pickle', 'rb')
        classifier = pickle.load(open_file)
        open_file.close()

        # Other classifiers
        classifiers = {'NaiveBayesClassifier': classifier}
        for classifier_class in self.classifier_classes:
            open_file = open('pickled_files/' + classifier_class.__name__ + '.pickle', 'rb')
            classifier = pickle.load(open_file)
            open_file.close()

        self.classifiers = list(classifiers.values())

    def train(self, documents, all_words, num_features=5000):
        # Save documents
        save_file = open('pickled_files/documents.pickle', 'wb')
        pickle.dump(documents, save_file)
        save_file.close()

        # Get word features
        all_words = [w.lower() for w in word_tokenize(pos_file) + word_tokenize(neg_file)]
        word_dist = nltk.FreqDist(all_words)
        self.word_features = list(word_dist.keys())[:num_features]

        # Save word features
        save_file = open('pickled_files/word_features.pickle', 'wb')
        pickle.dump(word_features, save_file)
        save_file.close()

        featuresets = [(self.find_features(rev), category) for (rev, category) in reviews]
        train_set, test_set = train_test_split(featuresets, test_size=0.3)

        # Normal Naive Bayes
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        print ("Normal Naive Bayes Accuracy:", nltk.classify.accuracy(classifier, test_set))
        # classifier.show_most_informative_features(10)

        # Save Naive Bayes algorithm
        save_file = open('pickled_files/NaiveBayesClassifier.pickle', 'wb')
        pickle.dump(classifier, save_file)
        save_file.close()        

        # Various types of classifiers
        classifiers = {'NaiveBayesClassifier': classifier}
        for classifier_class in self.classifier_classes:
            # Train and store classifier
            classifier = SklearnClassifier(classifier_class())
            classifier.train(train_set)
            classifiers[classifier_class.__name__] = classifier
            print (classifier_class.__name__, ' Accuracy: ', nltk.classify.accuracy(classifier, test_set))

            # Save Naive Bayes algorithm
            save_file = open('pickled_files/' + classifier_class.__name__ + '.pickle', 'wb')
            pickle.dump(classifier, save_file)
            save_file.close()

        self.classifiers = list(classifiers.values())
        print('Voting Classifier Accuracy: ', nltk.classify.accuracy(self, test_set))

        # Precision and Recall
        predicted = {'pos': set(), 'neg': set()}
        actual = {'pos': set(), 'neg': set()}
        for i, (features, label) in enumerate(test_set):
             actual[self.classify(features)].add(i)
             predicted[label].add(i)

        print('\nPrecision & Recall')
        print('Positive Precision: ', precision(predicted['pos'], actual['pos']))
        print('Positive Recall: ', recall(predicted['pos'], actual['pos']))
        print('Negative Precision: ', precision(predicted['neg'], actual['neg']))
        print('Negative Recall: ', recall(predicted['neg'], actual['neg']))

    def find_features(self, document):
        words = word_tokenize(document)
        return {w: w in words for w in self.word_features}


def train_vote_classifier():
    pos_file = open('short_reviews/positive.txt', 'r', encoding='latin-1').read()
    neg_file = open('short_reviews/negative.txt', 'r', encoding='latin-1').read()

    reviews = [(r, 'pos') for r in pos_file.split('\n')] + [(r, 'neg') for r in neg_file.split('\n')]
    all_words = [w.lower() for w in word_tokenize(pos_file) + word_tokenize(neg_file)]

    vote_classifier = VoteClassifier(MultinomialNB,
                                             BernoulliNB,
                                             LogisticRegression,
                                             SGDClassifier,
                                             SVC,
                                             LinearSVC,
                                             NuSVC)

    return vote_classifier.train(reviews, all_words)

    
