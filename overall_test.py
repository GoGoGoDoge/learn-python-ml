##!/usr/bin/python
import re
from pyclustering.samples.definitions import SIMPLE_SAMPLES, FCPS_SAMPLES;

from pyclustering.cluster import cluster_visualizer;
# from pyclustering.cluster.xmeans import xmeans, splitting_type;
from kmeans_fgy import kmeans
from pyclustering.utils import read_sample, timedcall;
import math

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

print(labels)
print("============================================")
# print(gram_mat_func)
gram_num = [[0 for a in range(d)] for b in range(d)]
# print(gram_num)
for i in range(d):
    for j in range(d):
        gram_num[i][j] = float(get_elmnt(i, j).subs([(x,0.5), (y,0.3)]))



def innerP2distance(gm_):
    l = len(gm_[0])
    dm_ = [[0 for x in range(l)] for y in range(l)]
    for i in range(l):
        for j in range(l):
            dm_[i][j] = math.sqrt(gm_[i][i] + gm_[j][j] - 2 * gm_[i][j])

    return dm_

dm = innerP2distance(gram_num)
print(dm)

# xmeans_instance = xmeans(gram_matrix, None, 20, 0.025, splitting_type.BAYESIAN_INFORMATION_CRITERION, False);
kmeans_instance = kmeans(dm, None, 10, 0.025);
(ticks, result) = timedcall(kmeans_instance.process);

clusters = kmeans_instance.get_clusters();
centers = kmeans_instance.get_centers();

print ("finish...")
print ("centers:", centers)
print ("clusters:", clusters)
print("clusters[1][2]: ", clusters[1][2])
print("clusters[1][3]: ", clusters[1][3])

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
    if nPos > nNeg:
        cluster_labels[i] = 1
    else:
        cluster_labels[i] = -1

print(cluster_labels)
