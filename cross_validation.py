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
import math
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

def innerP2distance(gm_):
    l = d
    dm_ = [[0 for x in range(l)] for y in range(l)]
    for i in range(l):
        for j in range(l):
            dm_[i][j] = math.sqrt(gm_[i][i] + gm_[j][j] - 2 * gm_[i][j])

    return dm_



def partition2train(dm_, i_):
    if i_ < cv-1:
        dm_train_ = [[0 for x in range(train_len)] for y in range(train_len)]
    else:
        dm_train_ = [[0 for x in range(d-remain_portion_len)] for y in range(d-remain_portion_len)]

    testing_indices_ = get_testing_indices(i_)
    lt = len(testing_indices_)
    iii = 0
    jjj = 0
    for ii in range(d):
        if ii < testing_indices_[0] or ii > testing_indices_[lt-1]:
            for jj in range(d):
                if jj < testing_indices_[0] or jj > testing_indices_[lt-1]:
                    dm_train_[iii][jjj] = dm[ii][jj]
                    jjj = jjj + 1
            iii = iii + 1
    return dm_train_

def get_testing_indices(i_):
    index_start = i_*len_portion
    if i_ < cv-1:
        index_end = index_start + len_portion # this last index is not used it is [) range
    else:
        index_end = d

    return range(index_start, index_end) # start <= idx <= end-1

def neg_cv_score(alpha=1., beta=0., k=5):
    '''
    # Performs cross validation on a gram matrix of training data and
    # returns the averaged accuracy scores.
    # The gram matrix 'gm' is generated from 'get_elmnt'.
    # The number of folds is specified by the variable 'cv'.
    '''
    print("This is neg_cv_score...")
    gm = [[0 for aa in range(d)] for bb in range(d)]
    for i in range(d):
        for j in range(d):
            gm[i][j] = float(get_elmnt_and_sub(i, j, alpha, beta))
            # gm[i][j] = float(get_elmnt(i, j).subs([(x,alpha), (y,beta)]))
            # gm[i][j] = get_elmnt(i, j).subs([(x,alpha), (y,beta)])
    print(gm)
    dm = innerP2distance(gm) # this is the pairwise distance matrix
    confusion_mat = {}
    confusion_mat_sum = [[0,0],[0,0]]
    for i in range(0,cv):
        dm_train = partition2train(dm, i) # a sub matrix extracted from the dm
        test_indices = get_testing_indices(i) # array of indices of the testing data points
        confusion_mat[i] = [[0,0],[0,0]] # [ [TN, FP], [FN, TP] ]

        # do clustering using the assigned k, output: list of {set of indices belonging to same cluster}
        kmeans_instance = kmeans(dm_train, None, k, 0.025)
        clusters = kmeans_instance.get_clusters()
        centers = kmeans_instance.get_centers()

        # use known labels to vote for the label of the cluster
        nClusters = len(centers)
        cluster_labels = [0 for x in range(nClusters)]
        for i in range(0, nClusters):
            nPoints = len(clusters[i][:])
            nPos = 0
            nNeg = 0
            for j in range(0, nPoints):
                if labels[clusters[i][j]] == "+1":
                    nPos = nPos + 1
                else:
                    nNeg = nNeg + 1
            print("For i th cluster, +1 v.s. -1 is: ", i, nPos, nNeg)
            if nPos > nNeg:
                cluster_labels[i] = 1
            else:
                cluster_labels[i] = -1

        print(cluster_labels)

        # do testing by finding the nearest cluster for the remaining points
        for test_i in test_indices:
            min_distance = numpy.Inf
            min_dist_cluster_idx = -1
            dist_2_cluster = 0
            for i in range(0, nClusters):
                cur_cluster = clusters[i][:];
                nPointsInCluster = len(cur_cluster)
                for j in range(0, nPointsInCluster):
                    tmp = dm[test_i][cur_cluster[j]]
                    tmp = tmp*tmp
                    dist_2_cluster = dist_2_cluster + tmp
                dist_2_cluster = math.sqrt(dist_2_cluster)
                dist_2_cluster = dist_2_cluster/nPointsInCluster

                if dist_2_cluster < min_distance:
                    min_distance = dist_2_cluster
                    min_dist_cluster_idx = i
            # determine whether this test data is fp, fn, tp, tn and add to the confusion_mat
            # TN = confusion_mat[i][0][0]
            # FP = confusion_mat[i][0][1]
            # FN = confusion_mat[i][1][0]
            # TP = confusion_mat[i][1][1]
            if labels[test_i] == "+1":
                if cluster_labels[min_dist_cluster_idx] == "+1":
                    confusion_mat[i][1][1] = confusion_mat[i][1][1] + 1
                else:
                    confusion_mat[i][1][0] = confusion_mat[i][1][0] + 1
            else:
                if cluster_labels[min_dist_cluster_idx] == "+1":
                    confusion_mat[i][0][1] = confusion_mat[i][0][1] + 1
                else:
                    confusion_mat[i][0][0] = confusion_mat[i][0][0] + 1

        # after all folds are done, add up the confusion_mat
        '''
        To obtain a “unified” matrix, you have only to perform addition of matrices.
        That is, individual fold confusion matrix are 2x2, and the unified confusion matrix is 2x2 as well.
        Note that every datum in a dataset is tested exactly one time through the entire folds,
        and hence, appears exactly one in some element of the unified matrix.
        Therefore, the unified confusion matrix looks as if the entire dataset were used as a test dataset.
        '''
        confusion_mat_sum = confusion_mat_sum + confusion_mat[i]
    # then compute the score using the combined confusion matrix, e.g. use accuracy.
    accuracy = (confusion_mat_sum[0][0]+confusion_mat_sum[1][1])/(confusion_mat_sum[0][0]+confusion_mat_sum[0][1]+confusion_mat_sum[1][0]+confusion_mat_sum[1][1])
    return -accuracy
'''
def neg_cv_score(x):
    # print('### neg_cv_score: {0}'.format(x))
    alpha, beta, k = x[:,0], x[:,1], x[:,2]
    n = x.shape[0]
    score = np.zeros(n)
    for i in range(n):
        score[i] = - cv_score(alpha[i], beta[i], k[i])
    return score
'''

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

len_portion = math.ceil(d/cv)
remain_portion_len = d%cv
train_len = (cv-1)*len_portion

domain=[{'name':'alpha', 'type':'continuous', 'domain':(0,1)},
        {'name':'beta', 'type':'continuous', 'domain':(0,1)},
        {'name':'k', 'type':'continuous', 'domain':(2,50)}]
#        {'name':'normal', 'type':'discrete', 'domain':(1,1)},
#        {'name':'kernel', 'type':'discrete', 'domain':(0,1)},
#        {'name':'gamma', 'type':'continuous', 'domain':(1.0e-3,1.0e3)}]
bo=GPyOpt.methods.BayesianOptimization(f=neg_cv_score,domain=domain)
# bo=GPyOpt.methods.BayesianOptimization(f=neg_cv_score,domain=domain,acquisition_type='LCB')
# bo.run_optimization(max_iter=30)
bo.run_optimization(max_iter=3)

bo.x_opt # Optimal solutions.
bo.fx_opt # Found minimum values.
