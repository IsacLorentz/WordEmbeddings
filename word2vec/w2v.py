import argparse
import os
import random
import re
import string
import time
from collections import defaultdict

import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

"""
This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2020 by Dmytro Kalpakchi.
"""


class Word2Vec(object):
    def __init__(
        self,
        filenames,
        dimension=300,
        window_size=2,
        nsample=10,
        learning_rate=0.025,
        epochs=5,
        use_corrected=True,
        use_lr_scheduling=True,
    ):
        """
        Constructs a new instance.
        
        :param      filenames:      A list of filenames to be used as the training material
        :param      dimension:      The dimensionality of the word embeddings
        :param      window_size:    The size of the context window
        :param      nsample:        The number of negative samples to be chosen
        :param      learning_rate:  The learning rate
        :param      epochs:         A number of epochs
        :param      use_corrected:  An indicator of whether a corrected unigram distribution should be used
        """
        self.__pad_word = "<pad>"
        self.__sources = filenames
        self.__H = dimension
        self.__lws = window_size
        self.__rws = window_size
        self.__C = self.__lws + self.__rws
        self.__init_lr = learning_rate
        self.__lr = learning_rate
        self.__nsample = nsample
        self.__epochs = epochs
        self.__nbrs = None
        self.__use_corrected = use_corrected
        self.__use_lr_scheduling = use_lr_scheduling

    def init_params(self, W, w2i, i2w):
        self.__W = W
        self.__w2i = w2i
        self.__i2w = i2w

    @property
    def vocab_size(self):
        return self.__V

    def clean_line(self, line):
        """
        The function takes a line from the text file as a string,
        removes all the punctuation and digits from it and returns
        all words in the cleaned line as a list
        
        :param      line:  The line
        :type       line:  str
        """

        line = re.sub(r"[^a-zA-Z\s]", "", line)
        line = line.strip().split()

        return line

    def text_gen(self):
        """
        A generator function providing one cleaned line at a time

        This function reads every file from the source files line by
        line and returns a special kind of iterator, called
        generator, returning one cleaned line a time.

        If you are unfamiliar with Python's generators, please read
        more following these links:
        - https://docs.python.org/3/howto/functional.html#generators
        - https://wiki.python.org/moin/Generators
        """
        for fname in self.__sources:
            with open(fname, encoding="utf8", errors="ignore") as f:
                for line in f:
                    yield self.clean_line(line)

    def get_context(self, sent, i):
        """
        Returns the context of the word `sent[i]` as a list of word indices
        
        :param      sent:  The sentence
        :type       sent:  list
        :param      i:     Index of the focus word in the sentence
        :type       i:     int
        """
        #
        # REPLACE WITH YOUR CODE
        #

        sentenceLen = len(sent)
        startI = max(0, i - self.__lws)
        endI = min(sentenceLen, i + self.__rws)

        validIndices = list(range(startI, i)) + list(range(i + 1, endI + 1))
        validIndices = [i for i in validIndices if i < sentenceLen]

        contextIndices = []
        for idx in validIndices:
            contextIndices.append(self.__w2i[sent[idx]])

        return contextIndices

    def skipgram_data(self):
        """
        A function preparing data for a skipgram word2vec model in 3 stages:
        1) Build the maps between words and indexes and vice versa
        2) Calculate the unigram distribution and corrected unigram distribution
           (the latter according to Mikolov's article)
        3) Return a tuple containing two lists:
            a) list of focus words
            b) list of respective context words
        """
        #
        # REPLACE WITH YOUR CODE
        self.__unigram_count = {}
        self.__unigram_prob = {}
        self.__unigram_prob_corr = {}
        self.__w2i = {}
        self.__i2w = []
        i = 0

        for line in self.text_gen():
            for word in line:
                if word not in self.__w2i:
                    self.__w2i[word] = i
                    self.__i2w.append(word)
                    self.__unigram_count[word] = 1
                    i += 1
                else:
                    self.__unigram_count[word] += 1

        noWords = len(self.__i2w)

        for key, val in self.__unigram_count.items():
            self.__unigram_prob[key] = val / noWords

        denom = sum([val ** 0.75 for _, val in self.__unigram_prob.items()])

        for key, val in self.__unigram_prob.items():
            self.__unigram_prob_corr[key] = (val ** 0.75) / denom

        # populate focus and context

        focusWords = []
        contextIndices = []
        for line in self.text_gen():
            for i, word in enumerate(line):
                if word not in focusWords:
                    focusWords.append(word)
                    contextIndices.append(self.get_context(line, i))
                if word in focusWords:
                    focusIndex = focusWords.index(word)
                    contextIndices[focusIndex].extend(self.get_context(line, i))

        return (focusWords, contextIndices)

    def sigmoid(self, x):
        """
        Computes a sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def sigmoidVec(self, z):

        vec = np.vectorize(self.sigmoid)
        return vec(z)

    def negative_sampling(self, number, xb, pos):
        """
        Sample a `number` of negatives examples with the words in `xb` and `pos` words being
        in the taboo list, i.e. those should be replaced if sampled.
        
        :param      number:     The number of negative examples to be sampled
        :type       number:     int
        :param      xb:         The index of the current focus word
        :type       xb:         int
        :param      pos:        The index of the current positive example
        :type       pos:        int
        """
        #
        # REPLACE WITH YOUR CODE
        #

        if self.__use_corrected:
            words = list(self.__unigram_prob.keys())
            probs = list(self.__unigram_prob.values())
        else:
            words = list(self.__unigram_prob_corr.keys())
            probs = list(self.__unigram_prob_corr.values())

        negSampleIndices = []
        it = 0

        while it < number:
            sampledIdx = self.__w2i[random.choices(population=words, weights=probs)[0]]
            if (
                sampledIdx != xb
                and sampledIdx != pos
                and sampledIdx not in negSampleIndices
            ):
                negSampleIndices.append(sampledIdx)
                it += 1

        return negSampleIndices

    def lrScheduler(self, alphaStart, alphaPred, nT, nWords, epochs):
        return (
            alphaStart * 1e-4
            if alphaPred < alphaStart * 1e-4
            else alphaStart * (1 - (nT / (epochs * nWords + 1)))
        )

    def train(self):
        """
        Performs the training of the word2vec skip-gram model
        """
        x, t = self.skipgram_data()
        N = len(x)
        self.__N = N
        print("Dataset contains {} datapoints".format(N))

        self.__W = np.random.normal(size=(N, self.__H))
        self.__U = np.random.normal(size=(N, self.__H))

        lr = self.__lr

        for ep in range(self.__epochs):
            for i in tqdm(range(N)):
                #
                # YOUR CODE HERE
                #

                lr = self.lrScheduler(self.__lr, lr, ep * N + i, N, self.__epochs)

                contextIndices = t[i]

                # update w.r.t positive samples
                for posIdx in contextIndices:

                    # update W w.r.t positive samples
                    self.__W[i, :] -= (
                        lr
                        * self.__U[posIdx, :]
                        * (self.sigmoidVec(self.__U[posIdx, :].T * self.__W[i, :]) - 1)
                    )

                    # update U w.r.t positive samples
                    self.__U[posIdx, :] -= (
                        lr
                        * self.__W[i, :]
                        * (self.sigmoidVec(self.__U[posIdx, :].T * self.__W[i, :]) - 1)
                    )

                    negSamples = self.negative_sampling(self.__nsample, i, posIdx)

                    for negIdx in negSamples:

                        # update W & U w.r.t negative samples.
                        self.__W[i, :] -= (
                            lr
                            * self.__U[negIdx, :]
                            * (self.sigmoidVec(self.__U[negIdx, :].T * self.__W[i, :]))
                        )

                        self.__U[negIdx, :] -= (
                            lr
                            * self.__W[i, :]
                            * (self.sigmoidVec(self.__U[negIdx, :].T * self.__W[i, :]))
                        )

    def find_nearest(self, words, metric="cosine", k=5):
        """
        Function returning k nearest neighbors with distances for each word in `words`
        
        We suggest using nearest neighbors implementation from scikit-learn 
        (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html). Check
        carefully their documentation regarding the parameters passed to the algorithm.
    
        To describe how the function operates, imagine you want to find 5 nearest neighbors for the words
        "Harry" and "Potter" using some distance metric `m`. 
        For that you would need to call `self.find_nearest(["Harry", "Potter"], k=5, metric='m')`.
        The output of the function would then be the following list of lists of tuples (LLT)
        (all words and distances are just example values):
    
        [[('Harry', 0.0), ('Hagrid', 0.07), ('Snape', 0.08), ('Dumbledore', 0.08), ('Hermione', 0.09)],
         [('Potter', 0.0), ('quickly', 0.21), ('asked', 0.22), ('lied', 0.23), ('okay', 0.24)]]
        
        The i-th element of the LLT would correspond to k nearest neighbors for the i-th word in the `words`
        list, provided as an argument. Each tuple contains a word and a similarity/distance metric.
        The tuples are sorted either by descending similarity or by ascending distance.
        
        :param      words:   Words for the nearest neighbors to be found
        :type       words:   list
        :param      metric:  The similarity/distance metric
        :type       metric:  string
        """
        #
        # REPLACE WITH YOUR CODE
        #
        # YOUR CODE HERE
        for word in words:
            if word not in self.__w2i:
                print("word not in text")
                return [None]
        wordVecs = [self.__W[self.__w2i[word]] for word in words]

        knn = NearestNeighbors(n_neighbors=k, metric=metric).fit(self.__W)

        outputList = []
        neighDist, neighIdx = knn.kneighbors(X=wordVecs)

        # loop over each query and get the corresponding neighbour from idx and the distance
        for i in range(neighIdx.shape[0]):
            outputList.append(
                [
                    (self.__i2w[neighIdx[i, j]], neighDist[i, j])
                    for j in range(neighIdx.shape[1])
                ]
            )
        return outputList

    def write_to_file(self):
        """
        Write the model to a file `w2v.txt`
        """
        try:
            with open("w2v.txt", "w") as f:
                W = self.__W
                f.write("{} {}\n".format(self.__N, self.__H))
                for i, w in enumerate(self.__i2w):
                    f.write(
                        w
                        + " "
                        + " ".join(map(lambda x: "{0:.6f}".format(x), W[i, :]))
                        + "\n"
                    )
        except:
            print("Error: failing to write model to the file")

    @classmethod
    def load(cls, fname):
        """
        Load the word2vec model from a file `fname`
        """
        w2v = None
        try:
            with open(fname, "r") as f:
                V, H = (int(a) for a in next(f).split())
                w2v = cls([], dimension=H)

                W, i2w, w2i = np.zeros((V, H)), [], {}
                for i, line in enumerate(f):
                    parts = line.split()
                    word = parts[0].strip()
                    w2i[word] = i
                    W[i] = list(map(float, parts[1:]))
                    i2w.append(word)

                w2v.init_params(W, w2i, i2w)
        except:
            print("Error: failing to load the model to the file")
        return w2v

    def interact(self):
        """
        Interactive mode allowing a user to enter a number of space-separated words and
        get nearest 5 nearest neighbors for every word in the vector space
        """
        print("PRESS q FOR EXIT")
        text = input("> ")
        while text != "q":
            text = text.split()
            neighbors = self.find_nearest(text, "cosine")

            for w, n in zip(text, neighbors):
                print("Neighbors for {}: {}".format(w, n))
            text = input("> ")

    def train_and_persist(self):
        """
        Main function call to train word embeddings and being able to input
        example words interactively
        """
        self.train()
        self.write_to_file()
        self.interact()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="word2vec embeddings toolkit")
    parser.add_argument(
        "-t",
        "--text",
        default="harry_potter_1.txt",
        help="Comma-separated source text files to be trained on",
    )
    parser.add_argument(
        "-s", "--save", default="w2v.txt", help="Filename where word vectors are saved"
    )
    parser.add_argument(
        "-d", "--dimension", default=50, help="Dimensionality of word vectors"
    )
    parser.add_argument("-ws", "--window-size", default=2, help="Context window size")
    parser.add_argument(
        "-neg", "--negative_sample", default=10, help="Number of negative samples"
    )
    parser.add_argument(
        "-lr", "--learning-rate", default=0.025, help="Initial learning rate"
    )
    parser.add_argument("-e", "--epochs", default=5, help="Number of epochs")
    parser.add_argument(
        "-uc",
        "--use-corrected",
        action="store_true",
        default=True,
        help="""An indicator of whether to use a corrected unigram distribution
                                for negative sampling""",
    )
    parser.add_argument(
        "-ulrs",
        "--use-learning-rate-scheduling",
        action="store_true",
        default=True,
        help="An indicator of whether using the learning rate scheduling",
    )
    args = parser.parse_args()

    if os.path.exists(args.save):
        w2v = Word2Vec.load(args.save)
        if w2v:
            w2v.interact()
    else:
        w2v = Word2Vec(
            args.text.split(","),
            dimension=args.dimension,
            window_size=args.window_size,
            nsample=args.negative_sample,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            use_corrected=args.use_corrected,
            use_lr_scheduling=args.use_learning_rate_scheduling,
        )
        w2v.train_and_persist()
