import sys, os, math
import numpy as np
import traceback
sys.path.append(os.getcwd())
import random

random.seed(42) 
np.random.seed(42)

# print([random.randint(0, 99) for _ in range(10)])
# print("inside test.py",random.random())

from lda import LDATopicModel
errs = 0
lda = LDATopicModel(K=3, dataset='animals')
lda.initialize_counts()

print("Checking vocabulary size...")
try:
    assert len(lda.vocabulary) == 29
    print('\tlooks good!')
except AssertionError:
    print('\tvocab size looks off.')
    errs += 1


print("Checking counts are initialized properly...")
try:
    
    assert lda.doc_lengths[17] == 6
    assert lda.doc_topic_counts[2][2] == 3
    assert lda.topic_counts[0] == 650
    assert lda.topic_word_counts[2]['cat'] == 31
    print('\tlooks good!')
except AssertionError:
    print('\tthere seem to be some problems with the initial counts.')
    print(lda.doc_lengths[17] == 6,6, lda.doc_lengths[17] )
    print(lda.doc_topic_counts[2][2] == 3,3, lda.doc_topic_counts[2][2] )
    print(lda.topic_counts[0] == 650,650,lda.topic_counts[0] )
    print(lda.topic_word_counts[2]['cat'] == 31, 31, lda.topic_word_counts[2]['cat'] )
    errs += 1

from contextlib import redirect_stdout
print('Training model...')
with redirect_stdout(None):
    lda.train()


print("Checking post-training counts...")
try:
    assert lda.doc_lengths[17] == 6
    assert lda.doc_topic_counts[2][2] == 0
    assert lda.topic_counts[0] == 655
    assert lda.topic_word_counts[2]['cat'] == 7
    print('\tlooks good!')
except AssertionError:
    print('\tthere seem to be some problems with the post-training counts.')
    errs += 1





print("Trying on larger dataset...")
lda = LDATopicModel(K=3, dataset='20_newsgroups_subset')
lda.initialize_counts()

print("Checking calculation of document-topic probability...")
try:
    assert math.fabs(lda.theta_d_i(42, 1) - 0.323809) < 0.001
    print('\tlooks good!')
except AssertionError:
    print('\tthere seem to be some problems with the calculations in self.theta_d_i.')
    errs += 1

print("Checking calculation of topic-word probability...")
try:
    assert math.fabs(lda.phi_i_v(0, 'space') - 0.009111) < 0.001
    assert math.fabs(lda.phi_i_v(2, 'sell') - 0.001948) < 0.001 
    print('\tlooks good!')
except AssertionError:
    print('\tthere seem to be some problems with the calculations in self.phi_i_v.')
    errs += 1


if errs == 0:
    print('All tests passed!\n')
else:
    print("Seems like there's still some work to do.")
