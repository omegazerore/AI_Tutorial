N = int(input())

for _ in range(N):
    poison = 0
    x, y, z, w, n, m = list(map(int, input().split()))
    map_ = {'0': 0,
            '1': x,
            '2': y,
            '3': -z,
            '4': -w}
    for carrot in list(input().split()):
        m -= poison * n
        if m < 0:
            print("bye~Rabbit")
            break
        m += map_[carrot]
        if m < 0:
            print("bye~Rabbit")
            break
        if carrot == '4':
            poison += 1
    if m >= 0:
        print(f"{m}g")


