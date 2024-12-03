"""
給你N個正整數, 試求哪幾個之和剛好為M, 印出所有合條件的解, 如有多組解, 請按由小到大的順序印出(格式可參考樣例輸出)
"""
# dynamic programming

from copy import copy
from collections import deque


def find_sum(residue, remaining_queue: deque):

    while remaining_queue:
        element = remaining_queue.popleft()
        if residue - element > 0:
            res = find_sum(residue - element, copy(remaining_queue))
            if res is not None:
                return [element] + res
        elif residue - element == 0:
            return [element]
        else:
            return None

    return None


n, m = map(int, input().split())

my_list = list(map(int, input().split()))

my_list.sort()

my_queue = deque()

for a in my_list:
    if a <= m:
        my_queue.append(a)

find_sum(m, my_queue)

if not flag:
    print("-1")