"""
排列數字一定很容易嗎

現在給你一堆數字

請你幫我排序
"""
while True:
    try:
        N = int(input())
    except EOFError:
        break

    my_list = []

    for _ in range(N):
        my_list.append(int(input()))

    my_list.sort()
    for i in my_list:
        print(i)

