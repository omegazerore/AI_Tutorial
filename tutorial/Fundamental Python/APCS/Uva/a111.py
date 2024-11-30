import math

while True:
    try:
        n = int(input())
    except EOFError:
        break
    if n == 0:
        break

    count_list = [(n-i) * (n-i) for i in range(0, n)]

    print(sum(count_list))