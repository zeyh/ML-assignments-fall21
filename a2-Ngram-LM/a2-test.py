import sys, os, math
sys.path.append(os.getcwd())
from language_modeling import NgramLanguageModel

ngram_lm = NgramLanguageModel()
print("Training LM...")
ngram_lm.train('samiam.train')

errors = 0

print("Checking counts...")
try:
    assert ngram_lm.unigram_counts['SAM'] == 17
    assert ngram_lm.unigram_counts['HAM'] == 9
    assert ngram_lm.bigram_counts['GREEN_EGGS'] == 8
    print("\tlooks good!")
except AssertionError:
    errors += 1
    print("\tthere seems to be some problem with the counts.")



print("Checking unigram prediction...")
try:
    val = ngram_lm.predict_unigram("SAM AND HAM IN A MOUSE HOUSE")
    assert math.fabs(32.37 + val) < 1
    print('\tlooks good!')
except AssertionError:
    errors += 1
    print('\tthere seems to be a problem with unigram prediction.')
    print('\texpected around -32.37, but got',val)

print("Checking bigram prediction...")
try:
    val = ngram_lm.predict_bigram("SAM AND HAM IN A MOUSE HOUSE")
    assert math.fabs(33.70 + val) < 1
    print('\tlooks good!')
except AssertionError:
    errors += 1
    print('\tthere seems to be a problem with bigram prediction.')
    print('\texpected around -33.70, but got',val)

print("Checking unigram perplexity calculation...")
try:
    val = ngram_lm.test_perplexity('samiam.test','unigram')
    v1 = math.fabs(59.42 - val) < 1
    v2 = math.fabs(53.15 - val) < 1
    if v1 or v2:
        print('\tlooks good!')
    else:
        assert 0 == 1
except AssertionError:
    errors += 1
    print('\tthere seems to be a problem with unigram perplexity.')
    print('\texpected around 53.15 or 59.42, but got',val)

print("Checking bigram perplexity calculation...")
try:
    val = ngram_lm.test_perplexity('samiam.test','bigram')
    assert math.fabs(16.69 - val) < 1
    print('\tlooks good!')
except AssertionError:
    errors += 1
    print('\tthere seems to be a problem with bigram perplexity.')
    print('\texpected around 16.59, but got',val)

if errors == 0:
    print("All tests passed!")
