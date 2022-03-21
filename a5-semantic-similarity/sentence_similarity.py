import csv, string
import numpy as np
from scipy.stats import spearmanr
from embeddings import Embeddings
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import math
import nltk
nltk.download('punkt')

def read_sts(infile = 'data/sts-dev.csv'):
    sts = {}
    for row in csv.reader(open(infile), delimiter='\t'):
        if len(row) < 7: continue
        val = float(row[4])
        s1, s2 = row[5], row[6]
        sts[s1, s2] = val / 5.0
    return sts

def calculate_sentence_embedding(embeddings, sent, weighted = False):
    """
    Calculate a sentence embedding vector.

    If weighted is False, this is the elementwise sum of the constituent word vectors.
    If weighted is True, multiply each vector by a scalar calculated
    by taking the log of its word_rank. The word_rank value is available
    via a dictionary on the Embeddings class, e.g.:
       embeddings.word_rank['the'] # returns 1

    In either case, tokenize the sentence with the `word_tokenize` function,
    lowercase the tokens, and ignore any words for which we don't have word vectors. 

    Parameters
    ----------
    sent : str
        A sentence for which to calculate an embedding.

    weighted : bool
        Whether or not to use word_rank weighting.

    Returns
    -------
    np.array of floats
        Embedding vector for the sentence.
    
    """
    sent_tokenize = word_tokenize(sent)
    embed = np.zeros(embeddings.vec_dim)
    for i,t in enumerate(sent_tokenize):
        try:
            weight = math.log(embeddings.word_rank[t.lower()]) if weighted else 1
            embed += np.asarray(embeddings.embeddings[t.lower()]) * weight
        except KeyError:
            print("KeyError: Cannot find ", t, " in the word embedding's file")
            pass
    return embed



def score_sentence_dataset(embeddings, dataset, weighted = False):
    """
    Calculate the correlation between human judgments of sentence similarity
    and the scores given by using sentence embeddings.

    Parameters
    ----------
    dataset : dictionary of the form { (sentence, sentence) : similarity_value }
        Dataset of sentence pairs and human similarity judgments.
    
    weighted : bool
        Whether or not to use word_rank weighting.

    Returns
    -------
    float
        The Spearman's Rho ranked correlation coefficient between
        the sentence emedding similarities and the human judgments.     
    """
    rho_vec1 = np.asarray(list(dataset.values()))
    rho_vec2 = np.zeros(len(dataset))

    for i,d in enumerate(dataset):
        vec1 = calculate_sentence_embedding(embeddings, d[0], weighted)
        vec2 = calculate_sentence_embedding(embeddings, d[1], weighted)
        rho_vec2[i] = embeddings.cosine_similarity(vec1, vec2)
    
    return spearmanr(rho_vec1, rho_vec2).correlation

if __name__ == '__main__':
    embeddings = Embeddings(glove_file = "data/glove_top50k_50d.txt")
    # sts = read_sts()
    
    # print('STS-B score without weighting:', score_sentence_dataset(embeddings, sts))
    # print('STS-B score with weighting:', score_sentence_dataset(embeddings, sts, True))
    
    val = np.mean(calculate_sentence_embedding(embeddings, 'Over, the rainbow?', weighted=False))
    print(val, math.fabs(val + 0.147621) < 0.0001)
    # val2 = np.mean(calculate_sentence_embedding(embeddings, 'Over, the rainbow?', weighted=True))
    # print(val2, math.fabs(val2 + 0.960183) < 0.0001)
    
    # test_dataset = {('this one', 'that one'): 0.6, ('all of them', 'none of them'): 0.2, ('many people say', 'sources indicate'): 0.8}
    # val = score_sentence_dataset(embeddings, test_dataset, weighted=True)
    # print(val)
    

