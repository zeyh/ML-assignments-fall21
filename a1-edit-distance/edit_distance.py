import numpy as np

def del_cost(string):
    return 1

def ins_cost(string):
    return 1

def sub_cost(c1, c2):
    if c1 == c2: 
        return 0
    else:
        return 2

def min_edit_distance(source, target, do_print_chart=False):
    """Compare `source` and `target` strings and return their edit distance with
    Levenshtein costs, according to the algorithm given in SLP Ch. 2, Figure 2.17.

    Parameters
    ----------
    source : str
        The source string.
    target : str
        The target string.

    Returns
    -------
    int
        The edit distance between the two strings.
    """
    
    '''
    Ref: https://web.stanford.edu/~jurafsky/slp3/2.pdf Figure 2.16
    function MIN-EDIT-DISTANCE(source, target) returns min-distance
        n←LENGTH(source)
        m←LENGTH(target)
        
        Create a distance matrix distance[n+1,m+1]
        # Initialization: the zeroth row and column is the distance from the empty string
        D[0,0] = 0
        for each row i from 1 to n do
            D[i,0]←D[i-1,0] + del-cost(source[i])
        for each column j from 1 to m do
            D[0,j]←D[0, j-1] + ins-cost(target[j])
        
        # Recurrence relation:
        for each row i from 1 to n do
            for each column j from 1 to m do
                D[i, j]←MIN( D[i−1, j] + del-cost(source[i]),
                    D[i−1, j−1] + sub-cost(source[i], target[j]),
                    D[i, j−1] + ins-cost(target[j]))
        
        # Termination
        return D[n,m]
    
    
    '''
    # >>> YOUR ANSWER HERE
    n = len(source)
    m = len(target)
    
    print("source: ", source, " #n ", n)
    print("target: ", target, " #m ", m)
    D = np.zeros((n+1, m+1),dtype='int')  #create a distance matrix distance[n+1, m+1]
    D[:, 0] = np.arange(0, n+1, 1) #fill first column
    D[0, :] = np.arange(0, m+1, 1) #fill first row

    # fill D from bottom-up
    for i in range(n+1):
        for j in range(m+1):
            if i == 0 or j == 0:
                pass #already initialized
            elif source[i-1] == target[j-1]: #same char, do nothing
                D[i][j] = D[i-1][j-1]
            else:
                D[i][j] = min(
                    D[i-1][j] + del_cost(source[i-1]), #delete
                    D[i][j-1] + ins_cost(target[j-1]), #insert   
                    D[i-1][j-1] + sub_cost(source[i-1], target[j-1]))  #substitute
                # alternatively...
                # D[i][j] = 1 + min(
                #     D[i-1][j], #delete
                #     D[i][j-1], #insert   
                #     1+D[i-1][j-1])  #substitute
            
    print(D)          
    return D[n][m]       
    # print(np.apply_along_axis( myfunction, axis=1, arr=D ))
    # >>> END YOUR ANSWER
   
def backtrace(str1, str2, D):
    pass
     
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) == 3:
        w1 = sys.argv[1]
        w2 = sys.argv[2]
    else:
        w1 = 'intention'
        w2 = 'execution'
    print('edit distance between', repr(w1), 'and', repr(w2), 'is', min_edit_distance(w1, w2))

    print("hello! for simple testing...")
    print(min_edit_distance("cat","hat"))

