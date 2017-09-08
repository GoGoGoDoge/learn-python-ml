#!/usr/bin/python
import re
from pyclustering.samples.definitions import SIMPLE_SAMPLES, FCPS_SAMPLES;

# from pyclustering.cluster import cluster_visualizer;
from kmeans_fgy import kmeans # , splitting_type
from pyclustering.utils import read_sample, timedcall;
import sympy
# from sklearn.model_selection import cross_val_score
# from sklearn import svm
# import numpy as np
import numpy
import GPy
import GPyOpt


def insert_ast(s):
    # Inserts asterisks * for sympy: xy -> x*y, 3x -> 3*x
    return re.sub(q, r'x*y', re.sub(p, r'\1*\2', s))

def set_elmnt(i, j, s):
    # Sets a SymPy object representing a string s as an (i, j) element
    gram_mat_func[(min([i,j]),max([i,j]))]=sympy.sympify(s)

def get_elmnt(i, j):
    # Returns an (i, j) element, if present, and 'None', otherwise.
    a = min([i,j])
    b = max([i,j])
    if (a,b) in gram_mat_func:
        return gram_mat_func[(a,b)]
    else:
        return None

def get_elmnt_and_sub(i, j, alpha, beta):
    # Returns an (i, j) element, if present, and 'None', otherwise.
    a = min([i,j])
    b = max([i,j])
    if (a,b) in gram_mat_func:
        return gram_mat_func[(a,b)].subs([(x,alpha), (y,beta)])
    else:
        return None

def innerP2distance(_gm):
    _dm = [[0 for aa in range(d)] for bb in range(d)]
    for i in range(d):
        for j in range(d):
            _dm[i][j] = numpy.sqrt(_gm[i][i] + _gm[j][j] - 2 * _gm[i][j])

    return _dm



def partition2train(_dm, _i):
    if _i < cv-1:
        _dm_train = [[0 for aa in range(d-len_portion)] for bb in range(d-len_portion)]
    else:
        _dm_train = [[0 for aa in range(d-remain_portion_len)] for bb in range(d-remain_portion_len)]

    _testing_indices = get_testing_indices(_i)
    lt = len(_testing_indices)
    iii = 0
    jjj = 0
    for ii in range(0,d):
        if ii < _testing_indices[0] or ii > _testing_indices[lt-1]:
            jjj = 0
            for jj in range(0,d):
                if jj < _testing_indices[0] or jj > _testing_indices[lt-1]:
                    _dm_train[iii][jjj] = _dm[ii][jj]
                    jjj = jjj + 1
                    #print("jjj: ", jjj)
            iii = iii + 1
            #print("iii: ", iii)
    return _dm_train

def get_testing_indices(_i):
    index_start = _i*len_portion
    if _i < cv-1:
        index_end = index_start + len_portion # this last index is not used it is [) range
    else:
        index_end = d

    return range(index_start, index_end) # start <= idx <= end-1

def cv_score(alpha=1., beta=0., k=5):
    '''
    # Performs cross validation on a gram matrix of training data and
    # returns the averaged accuracy scores.
    # The gram matrix 'gm' is generated from 'get_elmnt'.
    # The number of folds is specified by the variable 'cv'.
    '''
    print("This is neg_cv_score, alpha = ", alpha, "beta = ", beta, "k = ", k)
    gm = [[0 for aa in range(d)] for bb in range(d)]
    for i in range(d):
        for j in range(d):
            gm[i][j] = float(get_elmnt_and_sub(i, j, alpha, beta))
            # gm[i][j] = float(get_elmnt(i, j).subs([(x,alpha), (y,beta)]))
            # gm[i][j] = get_elmnt(i, j).subs([(x,alpha), (y,beta)])
    #print(gm)
    dm = innerP2distance(gm) # this is the pairwise distance matrix
    print( "size of dm:", len(dm[0][:]), len(dm) )
    confusion_mat = {}
    confusion_mat_sum = [[0,0],[0,0]]
    for i in range(0,cv):
        print("No. i fold: ", i)
        dm_train = partition2train(dm, i) # a sub matrix extracted from the dm
        print("size of dm_train: ", len(dm_train[0][:]), len(dm_train) )
        test_indices = get_testing_indices(i) # array of indices of the testing data points
        confusion_mat[i] = [[0,0],[0,0]] # [ [TN, FP], [FN, TP] ]

        # print(dm_train)
        # do clustering using the assigned k, output: list of {set of indices belonging to same cluster}

        #float_dm_train = []
        #for row in range(len(dm_train)):
        #    row_value = []
        #    for col in range(len(dm_train[0])):
        #        row_value.append(float(dm_train[row][col]))
        #    float_dm_train.append(row_value)

        #print(float_dm_train)
        kmeans_instance = kmeans(dm_train, None, 2*k, 0.025)

        #kmeans_instance = kmeans(float_dm_train, None, 2*k, 0.025)
        kmeans_instance.process()
        clusters = kmeans_instance.get_clusters()
        centers = kmeans_instance.get_centers()
        print("clusters: ", clusters)
        print("centers: ", centers)
        input("finished clustering, enter something...")
        #print("clusters[1][2]: ", clusters[1][2])
        #print("clusters[1][3]: ", clusters[1][3])
        # use known labels to vote for the label of the cluster
        nClusters = len(centers)
        print("Number of cluster is: ", nClusters)
        cluster_labels = [0 for x in range(nClusters)]
        for ic in range(0, nClusters):
            print("For the ith cluster: ", ic)
            nPoints = len(clusters[ic])
            print("     Number of points in current cluster: ", nPoints)
            nPos = 0
            nNeg = 0
            for jc in range(0, nPoints):
                if labels[clusters[ic][jc]] == "+1":
                    nPos = nPos + 1
                else:
                    nNeg = nNeg + 1
            print("For i th cluster, +1 v.s. -1 is: ", ic, nPos, nNeg)
            if nPos > nNeg:
                cluster_labels[ic] = 1
            else:
                cluster_labels[ic] = -1

        print("Labels of the clusters: ", cluster_labels)

        # do testing by finding the nearest cluster for the remaining points
        print("The test indices are: ", test_indices)
        for test_i in test_indices:
            min_distance = numpy.Inf
            min_dist_cluster_idx = -1
            dist_2_cluster = 0
            for ii in range(0, nClusters):
                cur_cluster = clusters[ii][:];
                nPointsInCluster = len(cur_cluster)
                for jj in range(0, nPointsInCluster):
                    tmp = dm[test_i][cur_cluster[jj]]
                    tmp = tmp*tmp
                    dist_2_cluster = dist_2_cluster + tmp
                dist_2_cluster = numpy.sqrt(dist_2_cluster)
                dist_2_cluster = dist_2_cluster/nPointsInCluster

                if dist_2_cluster < min_distance:
                    min_distance = dist_2_cluster
                    min_dist_cluster_idx = ii
            # determine whether this test data is fp, fn, tp, tn and add to the confusion_mat
            print(test_i)
            print("Label of this test point: ", labels[test_i])
            print("Idx of the nearest cluster: ", min_dist_cluster_idx)
            print("Label of the nearest cluster: ", cluster_labels[min_dist_cluster_idx])
            #print(confusion_mat[i])
            if labels[test_i] == "+1":
                if cluster_labels[min_dist_cluster_idx] == 1:
                    confusion_mat[i][1][1] = confusion_mat[i][1][1] + 1 # TP = confusion_mat[i][1][1]
                else:
                    confusion_mat[i][1][0] = confusion_mat[i][1][0] + 1 # FN = confusion_mat[i][1][0]
            else:
                if cluster_labels[min_dist_cluster_idx] == 1:
                    confusion_mat[i][0][1] = confusion_mat[i][0][1] + 1 # FP = confusion_mat[i][0][1]
                else:
                    confusion_mat[i][0][0] = confusion_mat[i][0][0] + 1 # TN = confusion_mat[i][0][0]

        print("Confusion mat of current fold: ", confusion_mat[i])
        input("enter sth to conntinue next fold...")
        # after all folds are done, add up the confusion_mat
        ##confusion_mat_sum = confusion_mat_sum + confusion_mat[i]
        for sum_i in range(0,2):
            for sum_j in range(0,2):
                confusion_mat_sum[sum_i][sum_j] = confusion_mat_sum[sum_i][sum_j] + confusion_mat[i][sum_i][sum_j]
        print("End of i th fold. ", i)
    # then compute the score using the combined confusion matrix, e.g. use accuracy.
    accuracy = (confusion_mat_sum[0][0]+confusion_mat_sum[1][1])/(confusion_mat_sum[0][0]+confusion_mat_sum[0][1]+confusion_mat_sum[1][0]+confusion_mat_sum[1][1])
    print("Final accuracy for this set of parameter is: ", accuracy)
    return accuracy

'''
To obtain a “unified” matrix, you have only to perform addition of matrices.
That is, individual fold confusion matrix are 2x2, and the unified confusion matrix is 2x2 as well.
Note that every datum in a dataset is tested exactly one time through the entire folds,
and hence, appears exactly one in some element of the unified matrix.
Therefore, the unified confusion matrix looks as if the entire dataset were used as a test dataset.
'''


def neg_cv_score(x):
    # print('### neg_cv_score: {0}'.format(x))
    alpha, beta, k = x[:,0], x[:,1], x[:,2]
    n = x.shape[0]
    score = numpy.zeros(n)
    print("n is: ", n)
    for i in range(n):
        print("PARAS: i, alpha, beta, k ", i, alpha[i], beta[i], k[i])
        score[i] = - cv_score(alpha[i], beta[i], k[i])
    return score


p=re.compile(r'(\d)([xy])')
q=re.compile(r'xy')


x,y=sympy.symbols("x y")

file_name = 'colon-cancer.kernel'
d = 134; # Dimension of the gram matrix = Number of samples
labels = {} # Key: Sample ID; Value: Class label
gram_mat_func = {} # Key: Pair of sample IDs; Value: SymPy object of kernel value


head_p=re.compile(r'(\d+):(\S+)?')
elmnt_p=re.compile(r'(\d+):(\S+)')

for line in open(file_name, 'r'):
    tokens = insert_ast(line.rstrip()).split()
    m = re.match(head_p, tokens[0])
    if m:
        g = m.groups()
        if g[1] != None:
            i = int(g[0])
            d = max([d,i+1])
            labels[i] = g[1]
        else:
            i = int(g[0])
            for t in tokens[1:]:
                g = re.match(elmnt_p, t).groups()
                j = int(g[0])
                set_elmnt(i,j,sympy.sympify(g[1]))

# The dictionary 'labels' is converted into a list object
labels=[labels[i] for i in range(0,d)]

'''
************************ End of kernel file processing *************************
'''

cv = 5

modulus = d % cv
len_portion = d / cv
if modulus > 0:
    len_portion = int(len_portion + 1)
# len_portion = int((float(d)/cv + 0.9999999) # e.g. 134/5=27
remain_portion_len = d - (cv-1)*len_portion # e.g. 26

alpha = 0.5
beta = 0.3
number_cluster = 5
#cv_score(alpha, beta, number_cluster)


domain=[{'name':'alpha', 'type':'continuous', 'domain':(0,1)},
        {'name':'beta', 'type':'continuous', 'domain':(0,1)},
        {'name':'k', 'type':'discrete', 'domain':(2,10)}]
#        {'name':'normal', 'type':'discrete', 'domain':(1,1)},
#        {'name':'kernel', 'type':'discrete', 'domain':(0,1)},
#        {'name':'gamma', 'type':'continuous', 'domain':(1.0e-3,1.0e3)}]
bo=GPyOpt.methods.BayesianOptimization(f=neg_cv_score,domain=domain)
    # bo=GPyOpt.methods.BayesianOptimization(f=neg_cv_score,domain=domain,acquisition_type='LCB')
    # bo.run_optimization(max_iter=30)
bo.run_optimization(max_iter=3)

bo.x_opt # Optimal solutions.
bo.fx_opt # Found minimum values.
print(x_opt)
