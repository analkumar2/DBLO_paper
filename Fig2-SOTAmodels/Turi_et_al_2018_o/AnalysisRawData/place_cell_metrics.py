def overall_mean(big_matrix,N):
    import numpy as np
    overallMean=0    
    for npyr in xrange(N):
        overallMean += np.mean(big_matrix[npyr,:,:])
    return overallMean

def spatial_info(rate_matrix, time_bin):
    import numpy as np
    
    # Time per bin
    tall = np.sum(time_bin)
    
    # Occupancy
    pall = time_bin/tall
    # Mean firing rate
    l = np.matmul(pall.T, rate_matrix).item()
    if l==0:
        l=1e-15

    ratio = rate_matrix/l
    ratio[ratio==0] = 1e-15 
                
    spatial_info = np.sum(np.multiply(pall, np.multiply(rate_matrix, np.log2(ratio))))         

    return spatial_info



def selectivity_index(rate_matrix):
    import numpy as np
    A = np.mean(rate_matrix)
    B = np.max(rate_matrix)
    if A == 0:
        A = 1e-15
       
    selectivity=float(B)/A

    return selectivity
    

def sparsity_index2(rate_matrix):
    # Size of field
    n1=rate_matrix.shape[0]
    n2=rate_matrix.shape[1]
    # Initialization
    sparsity_num = 0
    sparsity_den = 0
        
    for i in xrange(n1):
        for j in xrange(n2):
            
            # rate of each bin
            lx = rate_matrix[i,j]                
            # numerator
            sparsity_num += lx
            # denominator
            sparsity_den += lx**2            

    if sparsity_den == 0:
        sparsity_den = 1e-15
    L=n1     
    
    sparsity = 1 - (1.0/L)*(sparsity_num**2/float(sparsity_den))*(float(L)/(L-1))

    return sparsity

def peak_frequency(rate_matrix):
    import numpy as np
    return np.max(rate_matrix)


def field_size(rate_matrix, relfreq=1.0, track_length=200):
    import numpy as np
    if rate_matrix.shape[0] == 1 or rate_matrix.shape[1] == 1:
        peak=np.argmax(rate_matrix)
        
        # left from peak + peak
        counter1 = 0
        while rate_matrix[peak-counter1] >= relfreq:
            counter1 += 1
            if peak-counter1 < 0:
                counter1 -= 1
                break
        # right from peak    
        counter2 = 0
        while rate_matrix[peak+counter2]>=relfreq:
            counter2+=1
            if peak+counter2>=rate_matrix.shape[0]:
                counter2-=1
                break
            
        size = counter1+counter2
        size-=1
            
    else:
        pass
    
    if peak != 0 or peak != track_length:
        mean_in_place   = np.mean(rate_matrix[peak-(counter1-1):peak+counter2+1])
    elif peak==0:
        mean_in_place   = np.mean(rate_matrix[peak:peak+counter2+1])
    elif peak == track_length:
       mean_in_place = np.mean(rate_matrix[peak-(counter1-1):])
    
    if peak - counter1 == 0:
        mean_out_place1 = 0
    else:
        mean_out_place1 = np.mean(rate_matrix[:peak-(counter1-1)])
    if peak + counter2+1 == track_length:
        mean_out_place2 = 0 
    else:
        mean_out_place2 = np.mean(rate_matrix[peak+counter2+1:])
        
    mean_out_place  = np.mean([mean_out_place1, mean_out_place2])
    return size, mean_in_place, mean_out_place


def upper_tri_indexing(A):
    import numpy as np
    import sys
    '''
    A: input matrix
    returns its upper triangular elements
    excluding the diagonal.
    A should be square matrix
    return only upper tiangular elements, without diagonal
    '''

    if len(A.shape)!=2:
        sys.exit("Input is not two-dimensional.")        
     
    if (A.shape[0] != A.shape[1]):
        sys.exit("Matrix is not square.")
    
    m = A.shape[0]
    r,c = np.triu_indices(m,1)
    return A[r,c]


def stability_index(x, y=None):
    from place_cell_metrics import upper_tri_indexing
    import numpy as np
    
    if y is None:
        A = upper_tri_indexing(np.corrcoef(x.squeeze(),rowvar=1))
    else:
        A = upper_tri_indexing(np.corrcoef(x,y,rowvar=0))[0] 
    
    return np.tanh(A)
