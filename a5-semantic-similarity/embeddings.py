import math
import numpy as np
import heapq
from collections import Counter, deque

class Embeddings:

    def __init__(self, glove_file = '/projects/e31408/data/a5/glove_top50k_50d.txt'):
        self.embeddings = {}
        self.word_rank = {}
        self.vec_dim = 0
        for idx, line in enumerate(open(glove_file)):
            row = line.split()
            word = row[0]
            vals = np.array([float(x) for x in row[1:]])
            self.embeddings[word] = vals
            self.word_rank[word] = idx + 1
            self.vec_dim = self.embeddings[word].shape
        

    def __getitem__(self, word):
        return self.embeddings[word]

    def __contains__(self, word):
        return word in self.embeddings

    def vector_norm(self, vec):
        """
        Calculate the vector norm (aka length) of a vector.

        This is given in SLP Ch. 6, equation 6.8. For more information:
        https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm

        Parameters
        ----------
        vec : np.array
            An embedding vector.

        Returns
        -------
        float
            The length (L2 norm, Euclidean norm) of the input vector.
        """
        return np.linalg.norm(vec)

    def cosine_similarity(self, v1, v2):
        """
        Calculate cosine similarity between v1 and v2; these could be
        either words or numpy vectors.

        If either or both are words (e.g., type(v#) == str), replace them 
        with their corresponding numpy vectors before calculating similarity.

        Parameters
        ----------
        v1, v2 : str or np.array
            The words or vectors for which to calculate similarity.

        Returns
        -------
        float
            The cosine similarity between v1 and v2.
        """
        try:
            e1 = np.asarray(v1) if np.asarray(v1).shape == self.vec_dim else self.embeddings[v1]
            e2 = np.asarray(v2) if np.asarray(v2).shape == self.vec_dim else self.embeddings[v2]
            return np.dot(e1, e2) / (self.vector_norm(e1)* self.vector_norm(e2))
        except: #TODO very rough error handling
            print("Error! Please enter a valid word or vector")
            return 0.0

    def most_similar(self, vec, n = 5, exclude = []):
        """
        Return the most similar words to `vec` and their similarities. 
        As in the cosine similarity function, allow words or embeddings as input.


        Parameters
        ----------
        vec : str or np.array
            Input to calculate similarity against.

        n : int
            Number of results to return. Defaults to 5.

        exclude : list of str
            Do not include any words in this list in what you return.

        Returns
        -------
        list of ('word', similarity_score) tuples
            The top n results.        
        """
        max_heap = [] #heap sort and return the top n item in the heap
        for i,e in enumerate(self.embeddings):
            val = np.abs(self.cosine_similarity(vec, e))
            if e not in exclude:
                heapq.heappush(max_heap, (val, e))
            if len(max_heap) > n:
                heapq.heappop(max_heap)
                
        max_heap = sorted(max_heap, reverse=True) #sort from high to low
        return list(map(lambda t: (t[1],t[0]), max_heap)) #return only the string not the score
    

if __name__ == '__main__':
    embeddings = Embeddings(glove_file = "data/glove_top50k_50d.txt")
    top = embeddings.most_similar('goat', exclude='goat')
    print(top[0][0])