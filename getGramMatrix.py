import sys

data = sys.stdin.readlines()
N = len(data)
gram = []

for s1 in data:
    cur = []
    p1 = s1.split(" ")
    for s2 in data:
        p2 = s2.split(" ")
        #print p1[0], p1[1]
        cur.append(float(p1[0])*float(p2[0]) + float(p1[1])*float(p2[1]))
    gram.append(cur)

print gram
