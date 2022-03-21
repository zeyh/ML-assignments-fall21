from embeddings import Embeddings
from scipy import spatial
import math
import numpy as np

def read_turney_analogies(embeddings, path = 'data/SAT-package-V3.txt'):    
    questions = []
    letters = ['a','b','c','d','e']

    current = {'choices': []}
    start = True
    for line in open(path):
        if line.startswith('190 FROM REAL SATs') or line.strip() == '' or line.startswith('KS') or line.startswith('ML'): continue
        if len(line.strip()) == 1:
            current['answer'] = letters.index(line.strip())
            all_words = []
            all_words.extend(current['question'])
            for item in current['choices']:
                all_words.extend(item)
            if all(w in embeddings for w in all_words): 
                questions.append(current)
            current = {'choices':[]}
            start = True
            continue
        if start:
            current['question'] = tuple(line.split()[0:2])
            start = False
        else:
            current['choices'].append(tuple(line.split()[0:2]))
    return questions


def answer_by_analogy(embeddings, question, choices):
    """
    Answer an analogy question by the analogy (parallelogram) method.

    For a question a:b and possible choices of the form aa:bb,
    the answer is the one that maximizes cos(a - b + bb, aa).

    Parameters
    ----------
    question : tuple of (word, word)
       Words a and b to target.

    choices : list of tuples of (word, word)
       List of possible analogy matches aa and bb.

    Returns
    -------
    int
       index into `choices` of the estimated answer.
    """
    a = embeddings.embeddings[question[0]]
    b = embeddings.embeddings[question[1]]
    cos_val = []
    for i,c in enumerate(choices):
        vec1 = a-b+embeddings.embeddings[c[1]]
        vec2 = embeddings.embeddings[c[0]]
        cos_val.append(embeddings.cosine_similarity(vec1, vec2))
    # print(cos_val)
    max_val = max(cos_val)
    return cos_val.index(max_val)

def answer_by_parallelism(embeddings, question, choices):
    """
    Answer an analogy question by a parallelism method.

    For a question a:b and possible choices of the form aa:bb,
    the answer is the one that maximizes cos(a - b, aa - bb).

    Parameters
    ----------
    question : tuple of (word, word)
       Words a and b to target.

    choices : list of tuples of (word, word)
       List of possible analogy matches aa and bb.

    Returns
    -------
    int
       index into `choices` of the estimated answer.
    """
    a = embeddings.embeddings[question[0]]
    b = embeddings.embeddings[question[1]]
    cos_val = []
    for i,c in enumerate(choices):
        vec1 = a-b
        vec2 = embeddings.embeddings[c[0]]-embeddings.embeddings[c[1]]
        cos_val.append(embeddings.cosine_similarity(vec1, vec2))
    # print(cos_val)
    max_val = max(cos_val)
    return cos_val.index(max_val)

def evaluate(embeddings, dataset, method = answer_by_analogy):
    """
    Evaluate the guesses made by a given method.

    Parameters
    ----------
    dataset : list of dicts of the form {'question': (a, b), 'choices': [(aa, bb), ...], 'answer': idx}
        Represents a list of SAT analogy questions.

    method : func (either answer_by_analogy or answer_by_parallelism)
        The method to use. Note that in python you can pass functions
        along in this way without calling them, so inside this function
        you can call whichever method gets passed by doing `method(args)`.

    Returns
    -------
    float
        The accuracy of the given method: num_correct / num_total.
    """
    labels = np.zeros(len(dataset))
    predictions = np.zeros(len(dataset))
    count = 0
    right_ans_idx = []
    wrong_ans_idx = []
    for i,d in enumerate(dataset):
        labels[i] = d['answer']
        predictions[i] = method(embeddings, d['question'], d['choices'])
        if labels[i]  == predictions[i]:
            count += 1
            right_ans_idx.append(i)
        else:
            wrong_ans_idx.append(i)
    # print("y",labels)
    # print(predictions)
    # print(np.intersect1d(labels, predictions))
    return count/labels.shape[0], right_ans_idx, wrong_ans_idx


if __name__ == '__main__':
    embeddings = Embeddings(glove_file = "data/glove_top50k_50d.txt")
    SAT_questions = read_turney_analogies(embeddings,  path='data/SAT-package-V3.txt')
    test_q = SAT_questions[40]
    # a_answer = answer_by_analogy(embeddings, test_q['question'], test_q['choices'])
    # p_answer = answer_by_parallelism(embeddings, test_q['question'], test_q['choices'])
    
    # q_subset = SAT_questions[20:50]
    # analogy_result = evaluate(embeddings, q_subset, answer_by_analogy)
    # print(math.fabs(0.233333 - analogy_result) < 0.001, analogy_result)
    # parallelism_result = evaluate(embeddings, q_subset, answer_by_parallelism)
    # print(math.fabs(0.266666 - parallelism_result) < 0.001, parallelism_result)

    
    analogy_result,right_ans_idx1,wrong_ans_idx1 = evaluate(embeddings, SAT_questions, answer_by_analogy)
    parallelism_result,right_ans_idx2,wrong_ans_idx2 = evaluate(embeddings, SAT_questions, answer_by_parallelism)
    
    # common_true_idx = set(right_ans_idx1).intersection(set(right_ans_idx2))
    # for i in common_true_idx:
    #     print(SAT_questions[i]["question"], SAT_questions[i]["choices"][SAT_questions[i]["answer"]], " <<", SAT_questions[i]["choices"]  )
    
    # print()
    # print()
    # common_false_idx = set(wrong_ans_idx1).intersection(set(wrong_ans_idx2))
    # for i in common_false_idx:
    #     print(SAT_questions[i]["question"], SAT_questions[i]["choices"][SAT_questions[i]["answer"]], " <<", SAT_questions[i]["choices"]  )
    
    common_idx = set(right_ans_idx2).intersection(set(wrong_ans_idx1))
    for i in common_idx:
        print(SAT_questions[i]["question"], SAT_questions[i]["choices"][SAT_questions[i]["answer"]], " <<", SAT_questions[i]["choices"]  )
        
        
    print('Answering by analogy scored:',analogy_result)
    print('Answering by parallelism scored:',parallelism_result)
