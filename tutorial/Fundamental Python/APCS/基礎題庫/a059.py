"""
給你一個範圍 a 到 b ，請你找出 a 與 b 之間所有完全平方數的和。

例如：範圍 [3, 25] 中所有完全平方數的和就是 4 + 9 + 16 + 25  = 54

Given a range from
𝑎
a to b, find the sum of all perfect squares within this range.

For example, in the range

[3, 25]

the sum of all perfect squares is
4 + 9 + 16 + 25 = 54
2^2 + 3^2 + 4^2 + 5^2
"""
import math

rows = int(input())

for idx in range(rows):
    start_int = int(input())
    end_int = int(input())
    print("***********************************************")
    print(f"start_int: {start_int}; end_int: {end_int}")

    start_int = math.ceil(start_int**0.5)
    end_int = math.floor(end_int**0.5)
    print(f"start_int: {start_int}; end_int: {end_int}")
    print("***********************************************")

    result = 0
    for i in range(start_int, end_int+1):
        result += i**2

    print(f"Case {idx+1}: {result}")
    print("***********************************************")