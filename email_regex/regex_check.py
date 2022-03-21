
############################################
#
#   LING 334
#   Introduction to Computaional Linguistics
#   Spring 2021
#
#   In-class Regex Exercise!
#
############################################

"""

This is an exercise for trying to validate complicated email addresses with a regular expression.

To do this, edit the variable called 'YOUR_REGEX' below - this is all you have to edit in this file.
When you run the file it will show you which of the test cases (in valid.txt and invalid.txt) are
captured properly and not by your regex.

You're given a basic regex to start. My advice is to run this file and see what output it produces.
Pick one incorrect case, try to edit the regex to fix it, then run the file again. See which cases changed,

I would say this is quite tricky to get them all, and might require some advanced regex features like lookahead.
Getting 20 or more correct is really quite good.

The test cases are taken from here:
https://mkyong.com/regular-expressions/how-to-validate-email-address-with-regular-expression/

Once you've given it a really good go (at least 30 mins of trying), you can go look at their solution and 
see if you can use it to help fill in the remaining gaps.

"""

from collections import defaultdict
import re


# >>> YOUR ANSWER HERE

YOUR_REGEX = r'^.+@.+\..+[a-zA-Z]$'

# >>> END YOUR ANSWER


cases = defaultdict(set)
for ex in open('valid.txt'):
    email = ex.split('//')[0].strip()
    if re.match(YOUR_REGEX, email):
        cases["True Positive (valid and matched)"].add(ex.strip())
    else:
        cases["False Negative (valid and didn't match)"].add(ex.strip())

for ex in open('invalid.txt'):
    email = ex.split('//')[0].strip()
    if re.match(YOUR_REGEX, email):
        cases["False Positive (invalid and matched)"].add(ex.strip())
    else:
        cases["True Negative (invalid and didn't match)"].add(ex.strip())


print('-------- CORRECT ANSWERS ---------\n')
for typ in ['True Positive (valid and matched)', "True Negative (invalid and didn't match)"]:
    print('\t', typ)
    for ex in sorted(cases[typ]):
        print('\t\t', ex)
    print('')


print('\n-------- INCORRECT ANSWERS ---------\n')
for typ in ["False Positive (invalid and matched)", "False Negative (valid and didn't match)"]:
    print('\t', typ)
    for ex in sorted(cases[typ]):
        print('\t\t', ex)
    print('')

if len(cases['False Positive (invalid and matched)']) == 0 and len(cases["False Negative (valid and didn't match)"]) == 0:
    print('You got them all right!!! Wow!\n')
else:
    print("\nYou got {TPTN} correct, but didn't match {FN} cases where you should have, and incorrectly matched {FP} cases where you shouldn't have.\n".format(TPTN = len(cases['True Positive (valid and matched)']) + len(cases["True Negative (invalid and didn't match)"]), FN=len(cases["False Negative (valid and didn't match)"]), FP=len(cases['False Positive (invalid and matched)'])))
