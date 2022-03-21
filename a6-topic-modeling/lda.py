import enum
import random, string, nltk
from collections import Counter
import numpy as np
from itertools import chain
import math

from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import normalize


# Makes sure the process is always the same and reproducible for everyone,
# so we can compare results and run the autograder.
random.seed(42) 
np.random.seed(42)
# print("inside lda.py",random.random())

class LDATopicModel:

    def __init__(self, K=3, dataset='animals', iterations=100):
        # Hyperparameters
        self.num_topics = K
        self.topics = list(range(self.num_topics))
        self.iterations = iterations

        # symmetric, sparse dirichlet priors
        self.alpha = 1.0 / self.num_topics 
        self.beta = 1.0 / self.num_topics

        # Choose and load dataset
        if dataset == 'animals':
            self.documents = [[w.lower() for w in line.split()] for line in open('animals.txt')]
        elif dataset == '20_newsgroups_subset':
            self.documents = []
            for doc in fetch_20newsgroups(categories=('sci.space', 'misc.forsale', 'talk.politics.misc'), remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42, subset='train').data:
                tokenized = [w.lower() for w in word_tokenize(doc)]
                self.documents.append(tokenized)
        elif dataset == '20_newsgroups':
            self.documents = []
            for doc in fetch_20newsgroups(remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42, subset='train').data:
                tokenized = [w.lower() for w in word_tokenize(doc)]
                self.documents.append(tokenized)
        else:
            # As an extension, you could try loading in other datasets to run the model on.
            # You'll want to load in the documents as a list of lists, where each item in the
            # first list corresponds to the document, and each item within that corresponds
            # to a tokenized word.
            # >>> YOUR ANSWER HERE (optional)
            self.documents = []
            raise NotImplementedError # delete this if you add something here
            # >>> END YOUR ANSWER
        self.filter_docs()
        # print('# Docs:', len(self.documents))
        # print("inside lda.py",[random.randint(0, 99) for _ in range(10)])


    def filter_docs(self):
        """
        Filtering the available features for LDA makes a big difference on the
        quality of the output. This function is given for you and filters out
        the words in the documents based on document frequency and other
        factors. Check it out if you're interested.
        """
        # Calculate document frequencies
        dfs = Counter()
        for doc in self.documents:
            for word in set(doc):
                dfs[word] += 1

        # Remove stopwords, punctuation, numbers, and too common / rare words
        to_remove = set([l.strip() for l in open('english.stop')] + ["n't", "'s"])        
        for word in dfs:
            df_proportion = dfs[word] / len(self.documents)
            if df_proportion < 0.005: # remove words appearing in less than 0.5% of documents
                to_remove.add(word) 
            elif df_proportion > 0.5: # remove words appearing in more than 50% of documents
                to_remove.add(word)
            elif all(c in string.punctuation + string.digits for c in word): # remove punctuation and numerical
                to_remove.add(word)
            elif len(word) < 3: # remove very short words
                to_remove.add(word)

        # Re-constitute the dataset
        filtered_documents = []
        for doc in self.documents:
            new_doc = []
            for word in doc:
                if not word in to_remove:
                    new_doc.append(word)
            if len(new_doc) > 0:
                filtered_documents.append(new_doc)
        self.documents = filtered_documents
        return
            

    def decrement_counts(self, doc_id, word, z):
        """
        Decrement (-= 1) global counts for this assignment of (doc_id, word, z).
        
        Parameters
        ----------
        doc_id : int
            The index (identifier) of the current document.

        word : str
            The word identity of the current word.

        z: int
            The number of the current topic.


        Returns
        -------
        None (updates self.doc_topic_counts, self.topic_word_counts,
              self.topic_counts, and self.doc_lengths)
        """
        self.doc_topic_counts[doc_id][z] -= 1 
        self.topic_word_counts[z][word] -= 1
        self.topic_counts[z] -= 1
        self.doc_lengths[doc_id]  = self.doc_lengths[doc_id] - 1  # -1?
        
        # #make sure it is positive
        # self.doc_topic_counts[doc_id][z] = max(self.doc_topic_counts[doc_id][z], 0)
        # self.topic_word_counts[z][word] = max(self.topic_word_counts[z][word], 0)
        # self.topic_counts[z] = max(self.topic_counts[z], 0)
        # self.doc_lengths[doc_id]  = max(self.doc_lengths[doc_id] , 0)
        return


    def increment_counts(self, doc_id, word, z):
        """
        Increment (+= 1) global counts for this assignment of (doc_id, word, z).
        
        Parameters
        ----------
        doc_id : int
            The index (identifier) of the current document.

        word : str
            The word identity of the current word.

        z: int
            The number of the current topic.

        Returns
        -------
        None (updates self.doc_topic_counts, self.topic_word_counts,
              self.topic_counts, and self.doc_lengths)
        """
        self.doc_topic_counts[doc_id][z] += 1 
        self.topic_word_counts[z][word] += 1
        self.topic_counts[z] += 1
        self.doc_lengths[doc_id] = self.doc_lengths[doc_id]  + 1  
        return


    def initialize_counts(self):
        """
        Initialize all the counts we want to keep track of through
        the training process.
        
        This function should loop through every word in every document
        and make a random assignment from the possible topics
        to every word - this is most straightforward by doing:
            random.choice(self.topics)

        Once you've made that assignment, you'll need to update all of our
        counters so we can use them to calculate quantities of interest later.        
        Therefore I strongly recommend saving the randomly generated topic
        assignment for each word in a variable called, e.g., `z`, which you
        can use as a key in each of these dictionaries.

        Specifically, we need to track:

        - self.vocabulary, set of strings
            represents all words used in the corpus

        - self.doc_topic_counts, nested dictionary of the form:
            self.doc_topic_counts[doc_id][z] = number of times
                words assigned to topic `z` appear in document `doc_id`
        
        - self.topic_word_counts, nested dictionary of the form:
            self.topic_word_counts[z][word] = number of times
                word `word` has been assigned to topic `z`

        - self.doc_lengths, dictionary of the form:
            self.doc_lengths[doc_id] = count of total words in
                document `doc_id`
        
        - self.topic_counts, dictionary of the form:
            self.topic_counts[z] = count of how many words total
                have been assigned to topic `z`

        - self.assignments, dictionary of the form:
            self.assignments[doc_id, word_id] = z, keeping track of
               which topic every word in every document has been
               assigned to.       

        Make sure you think through and understand what each object
        is counting and why.

        Also take careful note of the python type and conceptual meaning
        of each item we are using as a key. You can call these other things,
        but using the variable names I've been using above they are:

        - doc_id : int, index of a document in self.documents
        - word_id : int, index of a given word within a given document,
            so self.documents[doc_id][word_id] would return the string
            of this word.
        - word : str, word type and therefore a member of self.vocabulary
        - z : int, topic id and therefore a member of self.topics
       

        Parameters
        ----------
        None

        Returns
        -------
        None (updates self.doc_topic_counts, self.topic_word_counts,
              self.topic_counts, self.doc_lengths, self.vocabulary,
              and self.assignments)
        """
        # print("word count",self.totalWordCount())
        
        # Collections of counts and assignments
        self.vocabulary = set(chain.from_iterable(self.documents))
        self.doc_topic_counts = { doc_id: Counter() for doc_id in range(len(self.documents)) }
        self.topic_word_counts = {z: Counter() for z in self.topics}
        self.doc_lengths = Counter()
        self.assignments = {}
        self.topic_counts = Counter()
        
        # Generate initial random topics and collect initial counts
        for i, d in enumerate(self.documents):
            tmp_assign = []
            for j,w in enumerate(d):
                z = random.choice(self.topics)
                tmp_assign.append([w, z])
                self.topic_word_counts[z][w] += 1
                self.doc_topic_counts[i][z] += 1
                self.assignments[i, j] = z
                self.topic_counts[z] += 1
            self.doc_lengths[i] = len(d)
            # self.assignments[i] = tmp_assign
        # self.topic_counts = sum(self.doc_topic_counts, Counter())
        return

    def totalWordCount(self):
        count = 0
        for d in self.documents:
            count += len(d)
        return count
                
    def print_topics(self):
        """
        Given for you, this function prints out the words most associated with each topic
        using your self.phi_i_v function.
        """
        print('---------')
        for z in self.topics:
            vals = {}
            for word in self.vocabulary:
                vals[word] = self.phi_i_v(z, word)
            print('TOPIC',z,': ', ' '.join(str(w) + ' ' + '{:.3f}'.format(v) for w, v in sorted(vals.items(), key=lambda x: x[1], reverse=True)[0:10]))
        print('---------')            
        return


    def theta_d_i(self, doc_id, z):
        """
        Calculate the current document weight for this topic.

        This is given in Applications of Topic Models Ch 1., pg. 15, equation 1.2.

        You should use the Dirichlet parameter given in self.alpha in this calculation.

        Parameters
        ----------
        doc_id : int
            The index (identifier) of the current document.

        z: int
            The number of the current topic.

        Returns
        -------
        float
            The document weight for this topic (probability of this topic given this document).
        """
        #how popular each topic in a document
        #N_{d,i} as the number of times document d uses topic i
        #Sum_k N_{d,k} the number of words in a document
        
        Ndi = self.doc_topic_counts[doc_id][z] + self.alpha
        Ndk_sumk = self.doc_lengths[doc_id] + self.alpha * self.num_topics
        # print(self.num_topics)
        # print(self.alpha)
        # print(self.doc_lengths[doc_id])

        # print("doc_topic_counts", self.doc_topic_counts[doc_id][z], self.doc_topic_counts[doc_id] )
        # print(self.assignments[doc_id], len(self.assignments[doc_id]))
        # print(self.documents[doc_id], len(self.documents[doc_id]))
        
        # print(self.doc_topic_counts[doc_id][z], "/", self.doc_lengths[doc_id])
        return Ndi/Ndk_sumk


    def phi_i_v(self, z, word):
        """
        Calculate the current topic weight for this word.

        This is given in Applications of Topic Models Ch 1., pg. 15, equation 1.3.

        You should use the Dirichlet parameter given in self.beta in this calculation.

        Parameters
        ----------
        z: int
            The number of the current topic.

        word : str
            The word identity of the current word.

        Returns
        -------
        float
            The topic weight for this word (probability of this word given this topic).
        """
        # the multinominal distribution of each topic over a word
        Viv = self.topic_word_counts[z][word] + self.beta 
        Viw_sumw = self.topic_counts[z] + self.beta * len(self.topic_word_counts[z]) # * len(self.vocabulary)
        return Viv/Viw_sumw

        
    def train(self):
        """
        Train the topic model using collapsed Gibbs sampling.

        Specifically, in each iteration of training, you should loop over
        each document in the corpus, and each word of each document.

        Then follow these steps:
        - Observe (and hold in a variable) the prior assignment of the
             current token.
        - Use your self.decrement_counts function to remove the current
             token from all our counts so it does not impact our 
             probability estimates.
        - Calculate the weight for each topic by multiplying together
             the results of self.theta_d_i and self.phi_i_v for the current
             document and word. This is equation 1.4 in Applications of 
             Topic Models, on pg. 16 in Ch. 1.
        - Use these weights to randomly sample a new topic for this token.
             Your probability of choosing the topic should be proportional
             to the weight calculated in the previous step.
             For consistency with the autograder I suggest using the 
             built-in `random` library's `random.choices` function,
             which takes a `weights` argument. Another option is numpy's
             `np.random.choice` function, but it's a bit more complicated
             so I don't particularly recommend it for this application. 
        - Update `self.assignments` with this new assignment, and use
             your self.increment_counts function to add this token back
             in to the counts with its new assignment.


        Parameters
        ----------
        None

        Returns
        -------
        None (updates self.assignments and other count dictionaries)
        """
        self.initialize_counts()
        # self.iterations = 1
        for iteration in range(self.iterations): #self.iterations
            if iteration % 10 == 0:
                print("\n\nIteration:", iteration)
                self.print_topics()
            else:
                print(iteration, end=' ', flush=True)

            # print(self.assignments[0])
            # self.documents = [self.documents[0]]
            # print(self.documents)
            for i,d in enumerate(self.documents):
                for j,w in enumerate(d):
                    z_prev = self.assignments[i, j]
                    self.decrement_counts(i, w, z_prev)
                    weights = np.asarray([self.theta_d_i(i,k) * self.phi_i_v(k,w) for k in self.topics])
                    # random.seed(42) 
                    # np.random.seed(42)
                    z_new = random.choices(self.topics, weights=weights)[0]
                    z_new_v2 = np.random.choice(self.topics, p=weights/sum(weights))
                    # print(z_new, weights)
                    self.assignments[i, j] = z_new
                    self.increment_counts(i, w, z_new)
                    # print(self.topic_counts)
            # print('\n\nTraining Complete')
        # self.print_topics()

    

if __name__ == '__main__':
    """
    Feel free to modify the below to play around with this as you are interested. 
    You can change 'K' to modify the number of learned topics, or 'dataset' to change which dataset to use.

    Available datasets already coded in include:
      - 'animals'                 a toy dataset of documents about three kinds of animals
      - '20_newsgroups_subset'    a subset of three categories of the classic '20 newsgroups' dataset
      - '20_newsgroups'           the entire 20 newsgroups; note this will take a relatively long time to run.
                                  this will also do better with more topics, since indeed it has more than 3.
                                  for me to run this dataset for 100 iterations with 20 topics takes about 40 minutes.
    """
    import sys, os, math
    import numpy as np
    import traceback
    # sys.path.append(os.getcwd())
    # random.seed(42) 
    # np.random.seed(42)
    print([random.randint(0, 99) for _ in range(10)])
    
    # lda = LDATopicModel(K=3, dataset='animals')
    # lda.initialize_counts()
    # from contextlib import redirect_stdout
    # print('Training model...')
    # with redirect_stdout(None):
    #     lda.train()
    # print(lda.doc_lengths[17] == 6, lda.doc_lengths[17])
    # print(lda.doc_topic_counts[2][2] == 3, lda.doc_topic_counts[2][2])
    # print(lda.topic_counts[0] == 650, lda.topic_counts[0])
    # print(lda.topic_word_counts[2]['cat'] == 31, lda.topic_word_counts[2]['cat'])
    
    
    
    # # ! ---------------------
    # lda0 = LDATopicModel(K=3, dataset='animals')
    # # lda0.initialize_counts()
    # # print(lda0.phi_i_v(0, 'pond'))
    # # print(lda0.theta_d_i(42, 1))
    # # print(lda0.topic_word_counts[2]['cat'] == 31,  "topic_word_counts",lda0.topic_word_counts[2]['cat'])
    # # print(lda0.topic_counts[0] == 650, "topic_counts", lda0.topic_counts[0])
    # # print(lda0.doc_topic_counts[2][2] == 3,"doc_topic_counts", lda0.doc_topic_counts[2][2])
    # # print(lda0.doc_lengths[17]==6, "doc_lengths", lda0.doc_lengths[17], sum(lda0.doc_lengths.values()))
    
    # {
    #     'gravity': 2, 
    #     'body': 2, 
    #     'system': 1, 
    #     'probe': 0, 
    #     'launched': 0, 
    #     'region': 0, 
    #     'affected': 0, 
    #     'lunar': 1, 
    #     'orbit': 2, 
    #     'large': 0, 
    #     'fuel': 0, 
    #     'slow': 2, 
    #     'idea': 2, 
    #     'objects': 2, 
    #     "'ll": 1, 
    #     'find': 1, 
    #     'trajectory': 2, 
    #     'makes': 2, 
    #     'nasa': 2, 
    #     'interested': 1, 
    #     'japan': 1, 
    #     'small': 2, 
    #     'hold': 1, 
    #     'lot': 0, 
    #     'issue': 2, 
    #     'news': 2, 
    #     'planetary': 1, 
    #     'report': 0, 
    #     'months': 2, 
    #     'ago': 2
    # }  
    
    # lda0.train()
    # print(lda0.doc_lengths[17] == 6, lda0.doc_lengths[17])
    # print(lda0.doc_topic_counts[2][2] == 3, lda0.doc_topic_counts[2][2])
    # print(lda0.topic_counts[0] == 650, lda0.topic_counts[0])
    # print(lda0.topic_word_counts[2]['cat'] == 31, lda0.topic_word_counts[2]['cat'])
    
    
    # # # ! --------------------
    # # print("--------------------")
    # # lda = LDATopicModel(K=3, dataset='20_newsgroups_subset')
    # # lda.initialize_counts()

    # # test2 = lda.phi_i_v(0, 'space')
    # # print(math.fabs(test2 - 0.009111) < 0.001, test2)
    # # test2 = lda.phi_i_v(2, 'sell')
    # # print(math.fabs(test2 - 0.001948) < 0.001, test2)
    
    # # print("--------------------")
    # # test = lda.theta_d_i(42, 1)
    # # print(math.fabs(test - 0.323809) < 0.001, test)
    
    
    # # lda.train()
    


                


