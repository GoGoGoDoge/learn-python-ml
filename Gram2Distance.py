import re

print("Calculate Distance")

#Example
gram_matrix = [["1.0x^3+1.0y^2+xy+6.0", "0x^3+2.0x^2y+3.0x^2"], ["0x^3+2.0x^2y+3.0x^2", "4.0x^2y+1.0x^2"]]

#Substitute given x and y
p=re.compile(r'(\d)([xy])')
q=re.compile(r'xy')
r=re.compile(r'\^')

t_gram_matrix = []
x_val = '2'
y_val = '3'
for i in range(len(gram_matrix)):
    t_gram_matrix.append([]);
    for j in range(len(gram_matrix[0])):
        res = re.sub(p, r'\1*\2', gram_matrix[i][j])
        res = re.sub(q, r'x*y', res);
        res = re.sub(r, r'**', res);
        res = re.sub(r'x', x_val, res);
        res = re.sub(r'y', y_val, res);
        t_gram_matrix[i].append(eval(res))
print t_gram_matrix
#Find the distance value for pairs (e.g. vector 0 and 1)
i = 0
j = 1
distance = t_gram_matrix[i][i] + t_gram_matrix[j][j] - 2 * t_gram_matrix[i][j]
print(distance)
