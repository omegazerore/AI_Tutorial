"""
給你N個正整數, 試求哪幾個之和剛好為M, 印出所有合條件的解, 如有多組解, 請按由小到大的順序印出(格式可參考樣例輸出)
"""
# dynamic programming

from copy import copy


flag = False

def find_sum(residue, answer, remaining_list):

    global flag

    copy_of_remaining_list = copy(remaining_list)

    for element in remaining_list:
        copy_of_remaining_list.pop(0)

        if residue - element > 0:
            find_sum(residue - element, answer + [element], copy_of_remaining_list)
        elif residue - element == 0:

            flag = True

            output = [str(a) for a in answer + [element]]

            print(" ".join(output))
        else:
            break


n, m = map(int, input().split())

my_list = list(map(int, input().split()))

my_list = [a for a in my_list if a <= m]

if len(my_list) == 0:
    flag = True
    print("-1")

if sum(my_list) < m:
    flag = True
    print("-1")

if sum(my_list) == m:
    flag = True
    print(" ".join([str(a) for a in my_list]))

my_list.sort()

find_sum(m, [], my_list)

if not flag:
    print("-1")







