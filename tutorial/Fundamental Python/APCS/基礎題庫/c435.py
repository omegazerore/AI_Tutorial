_ = input()

my_list = list(map(int, input().split()))

max_ = -200000

a_i = my_list[0]
a_j = my_list[1]

for k in my_list[2:]:
    if (a_i - k) > max_:
        max_ = a_i - k
    if k > a_i:
        a_i = k

print(max_)