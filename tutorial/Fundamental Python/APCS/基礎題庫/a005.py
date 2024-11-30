"""
Eva的家庭作業裏有很多數列填空練習。填空練習的要求是：已知數列的前四項，填出第五項。因 為已經知道這些數列只可能是等差或等比數列，她決定寫一個程式來完成這些練習。

Eva's homework includes many sequence fill-in-the-blank exercises. The task for these exercises is as follows: Given
the first four terms of a sequence, fill in the fifth term. Since it is already known that these sequences can only
be arithmetic or geometric, she decided to write a program to complete these exercises.
"""
n = int(input())

for _ in range(n):

    a, b, c, d = map(int, input().split())

    # test if arithmetic series

    answer = [a, b, c, d]

    if b - a == c - b == d - c:
        answer.append(d + (d - c))
    else:
        answer.append(d * (d // c))

    print(" ".join(map(str, answer)))