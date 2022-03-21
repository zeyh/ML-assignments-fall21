import sys, os, math
sys.path.append(os.getcwd())
from bias_audit import PMICalculator

errs = 0
print("Reading the corpora...")
all_labels_dev = PMICalculator(infile = '/projects/e31408/data/a4/snli_1.0/snli_1.0_dev.jsonl')

contra_dev = PMICalculator(infile = '/projects/e31408/data/a4/snli_1.0/snli_1.0_dev.jsonl', label_filter='contradiction')

print("Running preprocessing...")
all_labels_dev.preprocess()
contra_dev.preprocess()

print("Checking number of documents...")
try:
    assert all_labels_dev.n_docs == 10000
except AssertionError:
    print(f'\terror, on dev set with no label_filter expected n_docs == 10000 but got {all_labels_dev.n_docs}')
    errs += 1

try:
    assert contra_dev.n_docs == 3278
except AssertionError:
    print(f'\terror, on dev set with \'contradiction\' label_filter expected n_docs == 3278 but got {contra_dev.n_docs}')
    errs += 1
if errs == 0:
    print('\tlooks good!')

print("Checking vocabulary size...")

try:
    assert len(all_labels_dev.hypothesis_vocab_to_docs) == 720
    print('\tlooks good!')
except AssertionError:
    print(f'\terror, expected hypothesis vocab size of 720 but got {len(all_labels_dev.hypothesis_vocab_to_docs)}')
    errs += 1

print("Checking PMI calculation for zeroes...")

try:
    val = all_labels_dev.pmi('man','beast')
    assert val == 0.0   
    print('\tlooks good!')
except AssertionError:
    print(f"\terror, PMI for words that don\'t appear should be 0.0 but got {all_labels_dev.pmi('man','beast')}")
    errs += 1

print("Checking cross-analysis PMI calculation...")
try:
    val = contra_dev.pmi('boat','fish', cross_analysis=True)
    assert math.fabs(4.96435 - val) < 0.01
    print('\tlooks good!')
except AssertionError:
    print(f"\terror, in dev set contradictions expected PMI of ~4.96435 for 'boat' and 'fish' but got {contra_dev.pmi('boat','fish')}")
    errs += 1

print("Checking hypothesis-only PMI calculation...")
try:
    val = all_labels_dev.pmi('dog', 'frisbee', cross_analysis=False)
    assert math.fabs(3.6915 - val) < 0.01
    print('\tlooks good!')
except AssertionError:
    print(f"\terror, in dev set (all labels) expected PMI of ~3.6915 for 'dog' and 'frisbee' but got {contra_dev.pmi('dog','frisbee', cross_analysis=False)}")
    errs += 1

if errs == 0:
    print('All tests passed!\n')
