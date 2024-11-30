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

N = 500000
adjusted_N = int(math.ceil(math.sqrt(N)))
xy_set = set()

output_1 = 0
output_2 = set()
d = []

time_start = datetime.now()

for n in range(1, math.ceil(adjusted_N/math.sqrt(2))):
    for m in range(n+1, adjusted_N, 2):
        y = 2 * m * n
        x = m * m - n * n
        z = m * m + n * n
        if z > N:
            break
        if ((x*x + y*y) == z*z):
            x_base = x
            y_base = y
            z_base = z
            while z_base <= N:
                if x_base not in output_2:
                    output_2.add(x_base)
                if y_base not in output_2:
                    output_2.add(y_base)
                if z_base not in output_2:
                    output_2.add(z_base)
                # xy_set.add(tuple([x_base, y_base]))
                x_base += x
                y_base += y
                z_base += z
            if ((x % 2)==1):
                if (math.gcd(y, z)==1):
                    output_1 += 1
            else:
                if (math.gcd(x, z)==1):
                    output_1 += 1

print(output_1, N-len(output_2))

time_end = datetime.now()

print(time_end - time_start)

print()

# [3, 4, 5]
# [6, 8, 10]
# [12, 16, 20]
# [5, 12, 13]
# [7, 24, 25]
# [8, 15, 17]
# [9, 12, 15]
# [15, 20, 25]