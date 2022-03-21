import json, math
from collections import defaultdict, Counter, OrderedDict
import re 
from itertools import chain
import multiprocessing
import json
from decimal import Decimal
import argparse
import os
import sys

class PMICalculator:
    """Code to read the SNLI corpus and calculate PMI association metrics on it.
    """
    def __init__(self, infile = 'snli_1.0/snli_1.0_dev.jsonl', label_filter=None):
        self.infile = infile
        self.label_filter = label_filter # restricts the set of examples to read

        # mappings of words to indices of documents in which they appear
        self.premise_vocab_to_docs = defaultdict(set) 
        self.hypothesis_vocab_to_docs = defaultdict(set)
        self.n_docs = 0
        self.COUNT_THRESHOLD = 10
    
    def preprocess(self):
        """
        Read in the SNLI corpus and accumulate word-document counts to later calculate PMI.

        Your first task will be to look at the corpus and figure out how to read in its format.
        One hint - each line in the '.jsonl' files is a json object that can be read into a python
        dictionary with: json.loads(line)

        The corpus provides pre-tokenized and parsed versions of the sentences; you should use this
        existing tokenization, but it will require getting one of the parse representations and
        manipulating it to get just the tokens out. I recommend using the _binary_parse one.
        Remember to lowercase the tokens.

        As described in the assignment, instead of raw counts we will use binary counts per-document
        (e.g., ignore multiple occurrences of the same word in the document). This works well
        in short documents like the SNLI sentences.

        To make the necessary PMI calculations in a computationally efficient manner, the code is set up
        so that you do this slightly backwards - instead of accumulating counts of words, for each word
        we accumulate a set of indices (or other unique identifiers) for the documents in which it appears.
        This way we can quickly see, for instance, how many times two words co-occur in documents by 
        intersecting their sets. Document identifiers can be whatever you want; I recommend simply 
        keeping an index of the line number in the file with `enumerate` and using this.

        You can choose to modify this setup and do the counts for PMI some other way, but I do recommend
        going with the way it is.

        When loading the data, use the self.label_filter variable to restrict the data you look at:
        only process those examples for which the 'gold_label' key matches self.label_filter. 
        If self.label_filter is None, include all examples.        

        Finally, once you've loaded everything in, remove all words that don't appear in at least 
        self.COUNT_THRESHOLD times in the hypothesis documents.
        

        Parameters
        ----------
        None

        Returns
        -------
        None (modifies self.premise_vocab_to_docs, self.hypothesis_vocab_to_docs, and self.n_docs)

        """
        '''
        {
            "annotator_labels": ["neutral", "entailment", "neutral", "neutral", "neutral"], 
            "captionID": "4705552913.jpg#2", 
            "gold_label": "neutral", 
            "pairID": "4705552913.jpg#2r1n", 
            "sentence1": "Two women are embracing while holding to go packages.", 
            "sentence1_binary_parse": "( ( Two women ) ( ( are ( embracing ( while ( holding ( to ( go packages ) ) ) ) ) ) . ) )", 
            "sentence1_parse": "(ROOT (S (NP (CD Two) (NNS women)) (VP (VBP are) (VP (VBG embracing) (SBAR (IN while) (S (NP (VBG holding)) (VP (TO to) (VP (VB go) (NP (NNS packages)))))))) (. .)))", 
            "sentence2": "The sisters are hugging goodbye while holding to go packages after just eating lunch.", 
            "sentence2_binary_parse": "( ( The sisters ) ( ( are ( ( hugging goodbye ) ( while ( holding ( to ( ( go packages ) ( after ( just ( eating lunch ) ) ) ) ) ) ) ) ) . ) )", 
            "sentence2_parse": "(ROOT (S (NP (DT The) (NNS sisters)) (VP (VBP are) (VP (VBG hugging) (NP (UH goodbye)) (PP (IN while) (S (VP (VBG holding) (S (VP (TO to) (VP (VB go) (NP (NNS packages)) (PP (IN after) (S (ADVP (RB just)) (VP (VBG eating) (NP (NN lunch))))))))))))) (. .)))"
        }
        {
            "annotator_labels": ["entailment", "entailment", "entailment", "entailment", "entailment"], 
            "captionID": "4705552913.jpg#2", 
            "gold_label": "entailment", 
            "pairID": "4705552913.jpg#2r1e", 
            "sentence1": "Two women are embracing while holding to go packages.", 
            "sentence1_binary_parse": "( ( Two women ) ( ( are ( embracing ( while ( holding ( to ( go packages ) ) ) ) ) ) . ) )", 
            "sentence1_parse": "(ROOT (S (NP (CD Two) (NNS women)) (VP (VBP are) (VP (VBG embracing) (SBAR (IN while) (S (NP (VBG holding)) (VP (TO to) (VP (VB go) (NP (NNS packages)))))))) (. .)))", "sentence2": "Two woman are holding packages.", 
            "sentence2_binary_parse": "( ( Two woman ) ( ( are ( holding packages ) ) . ) )", "sentence2_parse": "(ROOT (S (NP (CD Two) (NN woman)) (VP (VBP are) (VP (VBG holding) (NP (NNS packages)))) (. .)))"
        }
        {
            "annotator_labels": ["contradiction", "contradiction", "contradiction", "contradiction", "contradiction"], 
            "captionID": "4705552913.jpg#2", 
            "gold_label": "contradiction", 
            "pairID": "4705552913.jpg#2r1c", 
            "sentence1": "Two women are embracing while holding to go packages.", 
            "sentence1_binary_parse": "( ( Two women ) ( ( are ( embracing ( while ( holding ( to ( go packages ) ) ) ) ) ) . ) )", 
            "sentence1_parse": "(ROOT (S (NP (CD Two) (NNS women)) (VP (VBP are) (VP (VBG embracing) (SBAR (IN while) (S (NP (VBG holding)) (VP (TO to) (VP (VB go) (NP (NNS packages)))))))) (. .)))", 
            "sentence2": "The men are fighting outside a deli.", 
            "sentence2_binary_parse": "( ( The men ) ( ( are ( fighting ( outside ( a deli ) ) ) ) . ) )", 
            "sentence2_parse": "(ROOT (S (NP (DT The) (NNS men)) (VP (VBP are) (VP (VBG fighting) (PP (IN outside) (NP (DT a) (NNS deli))))) (. .)))"
        }
        '''

        # read in the json data
        data = []
        for line in open(self.infile, 'r'):
            data.append(json.loads(line))
        # print(len(data),data[0])
        
        #words only
        sentences2 = []
        sentences1 = []
        
        #label_filter = contracdition
        for i,t in enumerate(data):
            if self.label_filter == None or (self.label_filter != None and self.label_filter == t['gold_label']):
                #for unigram
                tmp2 = t['sentence2'].lower()
                words_pattern = '([a-zA-Z]+)'
                extracted2 = re.findall(words_pattern, tmp2)
                sentences2.append(extracted2)
                
                tmp1 = t['sentence1'].lower()
                extracted1 = re.findall(words_pattern, tmp1)
                sentences1.append(extracted1)
                # print((self.label_filter != None and "contradiction" in t['annotator_labels']), t['annotator_labels'])
            # break
            
            #for bi-gram
            # tmp = t['sentence1_binary_parse'].lower()
            # words_pattern = '([a-zA-Z]+(\s*)[a-zA-Z]*)'
            # extracted = re.findall(words_pattern, tmp)
            # extracted = [a_tuple[0].strip() for a_tuple in extracted]
            # # extracted = list(filter(lambda x: (" " in x), extracted))
            # sentences.append(extracted)
        # print("!!!!!",len(sentences1), len(sentences2))

        self.n_docs = len(sentences1)    
        flatten_list1 = list(chain.from_iterable(sentences1))
        flatten_list2 = list(chain.from_iterable(sentences2))
        tmp_count1 = Counter({k: c for k, c in Counter(flatten_list1).items() if c >= self.COUNT_THRESHOLD})
        tmp_count2 = Counter({k: c for k, c in Counter(flatten_list2).items() if c >= self.COUNT_THRESHOLD})
        
        # print(len(tmp.keys()),tmp)
        print(">> Total Words with freq > thresh in Premise: ", len(tmp_count1.keys()))
        print(">> Total Words with freq > thresh in Hypothesis: ", len(tmp_count2.keys()))
        
        for i, w in enumerate(tmp_count1.keys()):
            for j, s in enumerate(sentences1):
                if w in s:
                    self.premise_vocab_to_docs[w].add(j)
        for i, w in enumerate(tmp_count2.keys()):
            for j, s in enumerate(sentences2):
                if w in s:
                    self.hypothesis_vocab_to_docs[w].add(j)
        
        #sort the hypothesis index list
        for i,k in enumerate(self.hypothesis_vocab_to_docs.keys()):
            self.hypothesis_vocab_to_docs[k] = set(sorted(list(self.hypothesis_vocab_to_docs[k])))
        print("✅ finished preprocessing ✅")
        # print(tmp_count1['the'])
        # print(len(self.premise_vocab_to_docs['the']),self.premise_vocab_to_docs['the'])
        # print(len(self.hypothesis_vocab_to_docs['the']),self.hypothesis_vocab_to_docs['the'])
        
        
    def pmi(self, word1, word2, cross_analysis=True):
        """
        Calculate the PMI between word1 and word1. the cross_analysis argument determines
        whether we look for word1 in the premise (True) or in the hypothesis (False).
        In either case we look for word2 in the hypothesis.        

        Since we are using binary counts per document, the PMI calculation is simplified.
        The numerator will be the number of total number of documents times the number
        of times word1 and word2 appear together. The denominator will be the number
        of times word1 appears total times the number of times word2 appears total.

        Do this using set operations on the document ids (values in the self.*_vocab_to_docs
        dictionaries). If either the numerator or denominator is 0 (e.g., any of the counts 
        are zero), return 0.

        Parameters
        ----------
        word1 : str
            The first word in the PMI calculation. In the 'cross' analysis type,
            this refers to the word from the premise.
        word2 : str
            The second word in the PMI calculation. In both analysis types, this
            is a word from the hypothesis documents.
        cross_analysis : bool
            Determines where to look up the document counts for word1;
            if True we look in the premise, if False we look in the hypothesis.

        Returns
        -------
        float
            The pointwise mutual information between word1 and word2.
        
        """
        # The numerator will be [the number of total number of documents] times the number
        # of times word1 and word2 appear together. The denominator will be the number
        # of times word1 appears total times the number of times word2 appears total.
        
        if not cross_analysis:
            set1 = self.hypothesis_vocab_to_docs[word1]
            set2 = self.hypothesis_vocab_to_docs[word2]
        else:
            set1 = self.premise_vocab_to_docs[word1]
            set2 = self.hypothesis_vocab_to_docs[word2]
        
        # print(set1, set2)
        # print("!!!!",len(set1), len(set2))
        intersection = set1.intersection(set2)
        numerator = self.n_docs * len(intersection)
        denominator = len(set1) * len(set2)
        # print("Intersection: ",len(intersection),self.n_docs * len(intersection))
        # print("✅ finished pmi calculation ✅ ",numerator,denominator )
        # print(self.n_docs, "*", len(intersection), "/", len(set1), "*", len(set2))
        # print(numerator,"/",denominator, "=",numerator/denominator)
        if denominator == 0 or numerator == 0:
            return 0.0
        return math.log2(numerator/denominator)

    def print_top_associations(self, target, n=10, cross_analysis=True):
        """
        Function to print the top associations by PMI across the corpus with
        a given target word. This is for qualitative use and 

        Since `word2` in the PMI calculation will always use counts in the 
        hypothesis, you'll want to loop over all words in the hypothesis vocab.

        Calculate PMI for each relative to the target, and print out the top n
        words with the highest values.
        """

        results = defaultdict()
        for _,w in enumerate(self.hypothesis_vocab_to_docs.keys()):
            # totalLen = len(self.hypothesis_vocab_to_docs[target])
            # if(i%100 == 0):
            #     print("progress: ", i, "/", totalLen)
            calculated = self.pmi(target, w, cross_analysis)
            # print(w, target, cross_analysis, calculated)
            results[w] = calculated
        # print(results)
        filtered_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True)[:n])
        writing = {target: filtered_results}
        return writing
        
        # print(printing)
        # print("✅ finished pmi calculation ✅ ",numerator,denominator )


def flatten_list(inputlist):
    return list(chain.from_iterable(inputlist))

def getTupleList(inputlist):
    return [a_tuple[0] for a_tuple in inputlist]

def setArgparse():
    parser = argparse.ArgumentParser(description='run PMI with the following specifications')
    parser.add_argument('--dir', default='snli_1.0/snli_1.0_dev.jsonl', help='input directory and name of training data')
    parser.add_argument('--filter', default=None,
                        help='[None, "neutral", "entailment", "contradiction"]')
    parser.add_argument('-ca','--ca', action='store_true', help='doing cross analysis?')
    # parser.add_argument('--thresh', default='10',
                        # help='a number that defines the COUNT_THRESHOLD to be filtered on')
    parser.add_argument('--topAssoc', default='10',
                        help='a number that defines top-n assiciations will be returned')
    parser.add_argument('--keyword', default='dog',
            help='a keyword to run top associations')
    parser.add_argument('--keywords',
            help='the txt contains keywords to run print_top_associations')
    parser.add_argument('--out', default='PMI_results.txt',
            help='output file directory and name')
    
    args = parser.parse_args()
    if args.filter not in ["neutral", "entailment", "contradiction"]:
        print("ERROR: --filter set to None because the input is not in [\"neutral\", \"entailment\", \"contradiction\"]")
        args.filter = None
    try:
        args.topAssoc = int(args.topAssoc)
    except:
        print("ERROR: please enter a valid positive integer")
        args.topAssoc = 10
    return args

def readtxt(fname):
    lines = []
    with open(fname) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def write2Json(writing, outName):
    out_file = open(outName, "w")
    printing = json.dump(writing, out_file, indent=4)
    out_file.close()


if __name__ == '__main__':
    print("Start testing...")
    args = setArgparse()
    print(args, args.dir, args.filter, args.ca, args.topAssoc)
    calculator = PMICalculator(args.dir, args.filter)
    calculator.preprocess()
    outList = []
    if args.keywords != None:
        print("📁 working with all the text in ", args.keywords)
        #read in the txt files
        keywords = readtxt(args.keywords)
        print(keywords)
        for k in keywords:
            outList.append(calculator.print_top_associations(k, args.topAssoc, args.ca))
    else:
        outList.append(calculator.print_top_associations(args.keyword, args.topAssoc, args.ca))
    
    if args.out == None or args.out == "":
        args.out = "results.json"
    write2Json(outList, args.out)
    
    
    
    
    




    # testing PMI
    # print(">> Testing PMI: ", all_labels.pmi('man','beast'))
    # # #3.6915
    # print(">> Testing PMI: ", all_labels.pmi('dog', 'frisbee', cross_analysis=False)) 
    #test contradiction
    # all_labels = PMICalculator(label_filter='contradiction')
    # all_labels.preprocess()
    # print(">> Testing PMI: ", all_labels.pmi('boat','fish', cross_analysis=True))


