#!/usr/bin/python
import re
from pyclustering.samples.definitions import SIMPLE_SAMPLES, FCPS_SAMPLES;

from pyclustering.cluster import cluster_visualizer;
# from pyclustering.cluster.xmeans import xmeans, splitting_type;
from xmeans import xmeans, splitting_type
from pyclustering.utils import read_sample, timedcall;
p=re.compile(r'(\d)([xy])')
q=re.compile(r'xy')

p=re.compile(r'(\d)([xy])')
q=re.compile(r'xy')

def insert_ast(s):
    # Inserts asterisks * for sympy: xy -> x*y, 3x -> 3*x
    return re.sub(q, r'x*y', re.sub(p, r'\1*\2', s))

import sympy
x,y=sympy.symbols("x y")

file_name = 'colon-cancer.kernel'
d = 134; # Dimension of the gram matrix = Number of samples
labels = {} # Key: Sample ID; Value: Class label
gram_mat_func = {} # Key: Pair of sample IDs; Value: SymPy object of kernel value

def set_elmnt(i, j, s):
    # Sets a SymPy object representing a string s as an (i, j) element
    gram_mat_func[(min([i,j]),max([i,j]))]=sympy.sympify(s)

def get_elmnt(i, j):
    # Returns an (i, j) element, if present, and 'None', otherwise.
    a=min([i,j])
    b=max([i,j])
    if (a,b) in gram_mat_func:
        return gram_mat_func[(a,b)]
    else:
        return None

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

from sklearn.model_selection import cross_val_score
from sklearn import svm
import numpy as np
import math

cv = 5

def innerP2distance(gm_):
    l = len(gm_[0])
    dm_ = [[0 for x in range(l)] for y in range(l)]
    for i in range(l):
        for j in range(ll):
            dm_[i][j] = math.sqrt(gm_[i][i] + gm_[j][j] - 2 * gm_[i][j])

    return dm_



def partition2train(gm_, i_, cv_):

def get_testing_indices(gm_, i_, cv_):

def neg_cv_score(alpha=1., beta=0., k=5):
    '''
    # Performs cross validation on a gram matrix of training data and
    # returns the averaged accuracy scores.
    # The gram matrix 'gm' is generated from 'get_elmnt'.
    # The number of folds is specified by the variable 'cv'.
    '''
    gm = [[get_elmnt(i,j).subs([(x,alpha),(y,beta)]) for j in range(d)] for i in range(d)]
    dm = innerP2distance(gm) # this is the pairwise distance matrix
    confusion_mat = {}
    confusion_mat_sum = [[0,0],[0,0]]
    for i in range(0,cv):
        dm_train = partition2train(dm, i, cv) # a sub matrix extracted from the dm
        indices_test = get_testing_indices(dm, i, cv) # array of indices of the testing data points
        confusion_mat[i] = [[0,0],[0,0]] # [ [TN, FP], [FN, TP] ]
        # do clustering using the assigned k, output: list of {set of indices belonging to same cluster}
            # should return e.g. cluster_indices_set

        # use known labels to vote for the label of the cluster

        # do testing by finding the nearest cluster for the remaining points
        for index in indices_test:
            for single_cluster[i] in cluster_indices_set:
                # sum of the squared distance from the test point to all points in the cluster
                avg_distance2cluster[i] = sum(...)
                avg_distance2cluster[i] = math.sqrt(avg_distance2cluster[i])
                avg_distance2cluster[i] /= number of points in the cluster

            # sort the avg_distance2cluster and find the shortest avg distance, get index of the nearest cluster

            # determine the label of the tested data by the label of the nearest cluster

            # determine whether this test data is fp, fn, tp, tn and add to the confusion_mat

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

import GPy
import GPyOpt
domain=[{'name':'alpha', 'type':'continuous', 'domain':(0,1)},
        {'name':'beta', 'type':'continuous', 'domain':(0,1)},
        {'name':'k', 'type':'continuous', 'domain':(2,50)}]
#        {'name':'normal', 'type':'discrete', 'domain':(1,1)},
#        {'name':'kernel', 'type':'discrete', 'domain':(0,1)},
#        {'name':'gamma', 'type':'continuous', 'domain':(1.0e-3,1.0e3)}]
bo=GPyOpt.methods.BayesianOptimization(f=neg_cv_score,domain=domain)
# bo=GPyOpt.methods.BayesianOptimization(f=neg_cv_score,domain=domain,acquisition_type='LCB')
bo.run_optimization(max_iter=30)

bo.x_opt # Optimal solutions.
bo.fx_opt # Found minimum values.
