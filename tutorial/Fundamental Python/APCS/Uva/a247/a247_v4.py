'''
while True:
    try:
        sentence = str(input())
    except EOFError:
        break
'''

import math
from datetime import datetime

N = 500000
adjusted_N = int(math.ceil(math.sqrt(N)))

output_1 = 0
output_2 = set()

time_start = datetime.now()

for n in range(1, math.ceil(adjusted_N/math.sqrt(2))):
    for m in range(n+1, adjusted_N, 2):
        y = 2 * m * n
        x = m * m - n * n
        z = m * m + n * n
        if z > N:
            break
        if math.gcd(x, y) != 1:
            continue
        multiplication = N//z + 1
        output_2.update([m * x for m in range(1, multiplication)])
        output_2.update([m * y for m in range(1, multiplication)])
        output_2.update([m * z for m in range(1, multiplication)])
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

# import math
#
# while True:
#     try:
#         N = int(input())
#     except EOFError:
#         break
#
#     adjusted_N = int(math.ceil(math.sqrt(N)))
#
#     output_1 = 0
#     output_2 = set()
#     d = []
#
#     for n in range(1, math.ceil(adjusted_N/math.sqrt(2))):
#         for m in range(n+1, adjusted_N, 2):
#             z = m * m + n * n
#             y = 2 * m * n
#             x = m * m - n * n
#             if z > N:
#                 break
#             if math.gcd(x, y) != 1:
#                 continue
#             multiplication = 1
#             while multiplication * z <= N:
#                 output_2.update({multiplication * x, multiplication * y, multiplication * z})
#                 multiplication += 1
#             if ((x % 2)==1):
#                 if (math.gcd(y, z)==1):
#                     output_1 += 1
#             else:
#                 if (math.gcd(x, z)==1):
#                     output_1 += 1
#
#     print(output_1, N-len(output_2))