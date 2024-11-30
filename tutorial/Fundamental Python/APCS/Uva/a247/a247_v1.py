'''
while True:
    try:
        sentence = str(input())
    except EOFError:
        break
'''

import math
from datetime import datetime

# N = int(in。u。t())

N = 500

output_1 = 0
output_2 = set()
ij_set = set()
d = []

upperlimit = N*N
i_upper = int(upperlimit/math.sqrt(2))


time_start = datetime.now()

for i in range(1, N):
    if i > i_upper:
        break
    for j in range(i+1, N+1):
        if j > i_upper:
            break
        if tuple([i, j]) in ij_set:
            continue
        summation =i*i + j*j
        if summation <= upperlimit:
            k = math.sqrt(summation)
            if k.is_integer():
                k = int(k)
                output_2.update(set([i, j, k]))
                ij_set.add(tuple([i, j]))
                # d.append([i, j, k])
                # extend to k close to boundary
                multiple_index = 2
                while k*multiple_index <= N:
                    ij_set.add(tuple([i * multiple_index, j*multiple_index]))
                    output_2.update(set([i*multiple_index, j*multiple_index, k*multiple_index]))
                    # d.append([i*multiple_index, j*multiple_index, k*multiple_index])
                    multiple_index += 1
                # print(i_set)
                # check GCD
                if (i % 2 == 0) and (j % 2 == 0):
                    continue
                if (math.gcd(i, j)!=1):
                    continue
                if (math.gcd(j, k)!=1):
                    continue
                if (math.gcd(i, k)==1):
                    output_1 += 1

print(output_1, N-len(output_2))

time_end = datetime.now()

print(time_end - time_start)

# print(d)
# 25
# 100
# 500
# 1000000
#
# 4 9
# 16 27
# 80 107
# 159139 133926
