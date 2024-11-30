'''
AC 33ms
'''

n, d = list(map(int, input().split()))

count = 0
cost = 0

for _ in range(n):
    prices = list(map(int, input().split()))

    if (max(prices) - min(prices)) >= d:
        count += 1
        cost += sum(prices)//3

print(count, cost)