import lxmls.readers.sentiment_reader as srs
import numpy as np
import scipy as scipy
import lxmls.classifiers.linear_classifier as lc
import sys
from lxmls.distributions.gaussian import *


class MultinomialNaiveBayes(lc.LinearClassifier):

    def __init__(self, xtype="gaussian"):
        lc.LinearClassifier.__init__(self)
        self.trained = False
        self.likelihood = 0
        self.prior = 0
        self.smooth = False
        self.smooth_param = 1

    def train(self, x, y):
        # n_docs = no. of documents
        # n_words = no. of unique words
        n_docs, n_words = x.shape

        # classes = a list of possible classes
        classes = np.unique(y)
        # n_classes = no. of classes
        n_classes = np.unique(y).shape[0]

        # initialization of the prior and likelihood variables
        prior = np.zeros(n_classes)
        likelihood = np.zeros((n_words, n_classes))

        # TODO: This is where you have to write your code!
        # You need to compute the values of the prior and likelihood parameters
        # and place them in the variables called "prior" and "likelihood".
        # Examples:
        # prior[0] is the prior probability of a document being of class 0
        # likelihood[4, 0] is the likelihood of the fifth(*) feature being
        # active, given that the document is of class 0
        # (*) recall that Python starts indices at 0, so an index of 4
        # corresponds to the fifth feature!

        # ----------
        # Solution to Exercise 1

        # P(x,y) --> probability of "fantastic" and positive occur at the same time in a document
        # P(y)   --> probability that a document is positive
        # P(x|y) --> probability of "fantastic" appearing in a document given that the document is positive
        # P(x,y) = P(y)P(x|y)

        # 1) prior: the class priors’ estimates are their relative frequencies

        # freq = np.bincount(y.flatten())  # also works, but generates a copy of y wasting memory
        freq = np.bincount(y[:, 0])
        prior[0] = freq[0] / n_docs
        prior[1] = freq[1] / n_docs

        # 2) likelihood: the class-conditional word probabilities are the relative frequencies of those words across documents with that class.

        # get negative and positive docs indices and filter docs

        idx0, vals0 = np.where(y == 0)
        idx1, vals1 = np.where(y == 1)

        docs0 = x[idx0, :]
        docs1 = x[idx1, :]

        # compute total of words on negative and positive documents

        num_words_in_docs0 = np.sum(docs0)
        num_words_in_docs1 = np.sum(docs1)

        # compute probability of each word for negative and positive documents

        # without smoothing --> problems with OOV words causing division by 0!
        # words_probs_docs0 = np.sum(docs0, 0) / num_words_in_docs0
        # words_probs_docs1 = np.sum(docs1, 0) / num_words_in_docs1

        # with smoothing
        words_probs_docs0 = (np.sum(docs0, 0) + 1) / (num_words_in_docs0 + n_words)
        words_probs_docs1 = (np.sum(docs1, 0) + 1) / (num_words_in_docs1 + n_words)

        likelihood[:, 0] = words_probs_docs0
        likelihood[:, 1] = words_probs_docs1


        # End solution to Exercise 1
        # ----------

        params = np.zeros((n_words+1, n_classes))
        for i in range(n_classes):
            params[0, i] = np.log(prior[i])
            params[1:, i] = np.nan_to_num(np.log(likelihood[:, i]))
        self.likelihood = likelihood
        self.prior = prior
        self.trained = True
        return params


if __name__ == '__main__':
    scr = srs.SentimentCorpus("books")
    mnb = MultinomialNaiveBayes()
    params_nb_sc = mnb.train(scr.train_X, scr.train_y)
    y_pred_train = mnb.test(scr.train_X, params_nb_sc)
    acc_train = mnb.evaluate(scr.train_y, y_pred_train)
    y_pred_test = mnb.test(scr.test_X, params_nb_sc)
    acc_test = mnb.evaluate(scr.test_y, y_pred_test)
    print("Multinomial Naive Bayes Amazon Sentiment Accuracy train: %f test: %f" % (acc_train, acc_test))
