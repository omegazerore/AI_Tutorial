import math

while True:
    try:
        n1, n2 = map(int, input().split())
    except EOFError:
        break

    if n1 > n2:
        print(int(n1) - int(n2))
    else:
        print(int(n2) - int(n1))



