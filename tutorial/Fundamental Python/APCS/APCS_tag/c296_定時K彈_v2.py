'''
約瑟夫環問題(Josephus problem)變體

https://en.wikipedia.org/wiki/Josephus_problem

f(N, M) = (f(N-1, M) + M)%N

AC 79ms
'''

N, M, K = map(int, input().split())

# Step 1: find the final position index of the lucky man

last_dead_man_index = 0

for number_of_survivers in range(N-K+1, N+1):
    last_dead_man_index = (last_dead_man_index + M) % (number_of_survivers)

print(last_dead_man_index + 1)





