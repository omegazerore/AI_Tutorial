'''
Lagrangian Multiplier

Solve the following equations:

2*A_1*X_1 + B_1 + lambda = 0
2*A_2*X_2 + B_2 + lambda = 0
X_1 + X_2 = n

AC 22ms 3.4MB
'''
import math


a_1, b_1, c_1 = list(map(int, input().split()))
a_2, b_2, c_2 = list(map(int, input().split()))
n = int(input())

'''
Sigma = x_1^2 * (a_1 + a_2) + x_1 * (b_1 - 2 * a_2 * n - b_2) + c_1 + a_2 * n * n + b_2 * n + c_2
'''

def summation(n_input):

    return a_1 * n_input * n_input + b_1 * n_input + c_1 + a_2 * (n-n_input) * (n-n_input) + b_2 * (n-n_input) + c_2

c = c_1 + a_2 * n * n + b_2 * n + c_2

if (a_1 + a_2) > 0:
    sigma = max(summation(n), summation(0))
elif (a_2 + a_2) == 0:
    b = b_1 - 2 * a_2 * n - b_2
    if b > 0:
        sigma = summation(n)
    elif b < 0:
        sigma = summation(0)
    else:
        sigma = c
else:
    n_k = (2 * a_2 * n + b_2 - b_1) / (2 * a_1 + 2 * a_2)
    if n_k > n:
        sigma = summation(n)
    elif n_k < 0:
        sigma = summation(0)
    else:
        sigma = max(summation(math.ceil(n_k)), summation(math.floor(n_k)))

print(sigma)