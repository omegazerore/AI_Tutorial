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

N = 25

output_1 = 0
output_2 = set()
ij_set = set()
d = []

upperlimit = N*N
i_upper = N/math.sqrt(2)

time_start = datetime.now()

for i in range(1, math.ceil(i_upper)):
    for j in range(i+1, N, 2):
        summation =i*i + j*j
        if summation <= upperlimit:
            k = math.sqrt(summation)
            if k.is_integer():
                k = int(k)
                # output_2.update(set([i, j, k]))
                i_base = i
                j_base = j
                k_base = k
                while k_base <= N:
                    print([i_base, j_base, k_base])
                    output_2.update(set([i_base, j_base, k_base]))
                    i_base += i_base
                    j_base += j_base
                    k_base += k_base
                if ((j % 2)==1):
                    if (math.gcd(j, k)==1):
                        output_1 += 1
                else:
                    if (math.gcd(i, k)==1):
                        output_1 += 1
        else:
            break

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
