##!/usr/bin/python
import re
from pyclustering.samples.definitions import SIMPLE_SAMPLES, FCPS_SAMPLES;

# from pyclustering.cluster import cluster_visualizer;
# from pyclustering.cluster.xmeans import xmeans, splitting_type;
from xmeans import xmeans, splitting_type
from pyclustering.utils import read_sample, timedcall;
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
#print(gram_mat_func)
gram_num = [[0 for a in range(d)] for b in range(d)]
# print(gram_num)
for i in range(d):
    for j in range(d):
        gram_num[i][j] = float(get_elmnt(i, j).subs([(x,0.5), (y,0.3)]))

print(gram_num)
# gram_num = [[42.529316490802, 42.717195285258, 41.805586172258, 42.978501640665996, 44.947537609731, 67.11094076958, 63.678011905026004, 66.310339525968, 66.346686765112, 64.240854664512], [42.717195285258, 42.98070563613, 41.977454326098, 43.247573480802, 45.215452860866996, 67.865308959276, 64.438999860762, 66.960868291632, 67.121517748872, 64.925375521176], [41.805586172258, 41.977454326098, 41.096366664404, 42.23355903671, 44.17077567328, 65.890462310272, 62.512220529489994, 65.12066685559199, 65.135130172156, 63.079009970274], [42.978501640665996, 43.247573480802, 42.23355903671, 43.516306488106, 45.495703276581, 68.304626633076, 64.858508648106, 67.389357355104, 67.55754427144, 65.343690739164], [44.947537609731, 45.215452860866996, 44.17077567328, 45.495703276581, 47.567561523780995, 71.351422915636, 67.743495902731, 70.412301958164, 70.56581610466999, 68.26513874790899], [67.11094076958, 67.865308959276, 65.890462310272, 68.304626633076, 71.351422915636, 108.703568183632, 103.419789987916, 106.82622402504, 107.64351116280801, 103.824638625396], [63.678011905026004, 64.438999860762, 62.512220529489994, 64.858508648106, 67.743495902731, 103.419789987916, 98.419459482706, 101.577857553744, 102.42836573036, 98.755881560544], [66.310339525968, 66.960868291632, 65.12066685559199, 67.389357355104, 70.412301958164, 106.82622402504, 101.577857553744, 105.098448426912, 105.748587762288, 102.077980935768], [66.346686765112, 67.121517748872, 65.135130172156, 67.55754427144, 70.56581610466999, 107.64351116280801, 102.42836573036, 105.748587762288, 106.60478522218399, 102.797937856836], [64.240854664512, 64.925375521176, 63.079009970274, 65.343690739164, 68.26513874790899, 103.824638625396, 98.755881560544, 102.077980935768, 102.797937856836, 99.18308837133]]

# xmeans_instance = xmeans(gram_matrix, None, 20, 0.025, splitting_type.BAYESIAN_INFORMATION_CRITERION, False);
xmeans_instance = xmeans(gram_num, None, 10, 0.025, splitting_type.BAYESIAN_INFORMATION_CRITERION, False);
(ticks, result) = timedcall(xmeans_instance.process);

clusters = xmeans_instance.get_clusters();
centers = xmeans_instance.get_centers();

print ("finish...")
print ("centers:", centers)
print ("clusters:", clusters)
