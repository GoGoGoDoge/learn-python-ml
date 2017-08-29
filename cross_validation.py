from sklearn.model_selection import cross_val_score
from sklearn import svm
import numpy as np
import math

cv = 5

def partition2train(gm_, i_, cv_):

def get_testing_indices(gm_, i_, cv_):

def cv_score(alpha=1., beta=0., k=5):
    # Performs cross validation on a gram matrix of training data and
    # returns the averaged accuracy scores.
    # The gram matrix 'gm' is generated from 'get_elmnt'.
    # The number of folds is specivied by the variable 'cv'.
    gm=[[get_elmnt(i,j).subs([(x,alpha),(y,beta)]) for j in range(d)] for i in range(d)]

    for i in range(0,cv):
        gm_train = partition2train(gm, i, cv) # a sub matrix extracted from the gm
        indices_test = get_testing_indices(gm, i, cv) # array of indices of the testing data points
        confusion_mat = [[0,0],[0,0]]
        # do clustering using the assigned k, output: list of {set of indices belonging to same cluster}
            # should return e.g. cluster_indices_set

        # use known labels to vote for the label of the cluster

        # do testing by finding the nearest cluster for the remaining points
        for index in indices_test:
            for single_cluster[i] in cluster_indices_set:
                # sum of the squared distance from the test point to all points in the cluster
                avg_distance2cluster[i] = sum(...)
                avg_distance2cluster[i] = sqrt(avg_distance2cluster[i])
                avg_distance2cluster[i] /= number of points in the cluster

            # sort the avg_distance2cluster and find the shortest avg distance, get index of the nearest cluster

            # determine the label of the tested data by the label of the nearest cluster

            # determine whether this test data is fp, fn, tp, tn and add to the confusion_mat

    # after all folds are done, add up the confusion_mat
    # then compute the score using the combined confusion matrix, e.g. use accuracy.


def neg_cv_score(x):
    # print('### neg_cv_score: {0}'.format(x))
    alpha, beta, k = x[:,0], x[:,1], x[:,2]
    n = x.shape[0]
    score = np.zeros(n)
    for i in range(n):
        score[i] = - cv_score(alpha[i], beta[i], k[i])
    return score

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
