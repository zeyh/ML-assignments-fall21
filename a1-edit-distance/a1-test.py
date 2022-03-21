import sys, os
sys.path.append(os.getcwd())
from edit_distance import min_edit_distance


items = [[('cat','hat'), 2,],
         [('inevitable', 'evidently'), 9],
         [('never','ever'), 1],
         [('ever','never'), 1],
         [('the same', 'the same'), 0],
         [('manifestly', 'magnificently'), 5],
         [('','hopefully'), 9]]
         
print('Checking answers:\n')
incorrect = 0
for pair, val in items:
    output = min_edit_distance(pair[0], pair[1])
    if output == val:
        print(repr(pair[0]), repr(pair[1]))
        print('\t', val, '\n')
    else:
        incorrect += 1
        print(repr(pair[0]), repr(pair[1]))
        print('\t', 'INCORRECT:', '\tshould have been',val, 'but you got',output,'\n')

if incorrect == 0:
    print('Everything looks good, well done!!')
else:
    print('You got',incorrect,'wrong - still some work to do.')
