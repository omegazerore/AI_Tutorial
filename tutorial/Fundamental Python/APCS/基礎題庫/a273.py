"""
有一個小朋友，站在第 n 階樓梯上，他一次能，而且只能，往下跳 k 個階梯，請問他能順利跳到第 0 階嗎？
舉例來說，小朋友一開始在第 n=9 階上，而他一次能往下跳 k=3 個階梯，那麼，他就會 9 -> 6 -> 3 -> 0 順利到達第 0 階。

A child is standing on the
𝑛
n-th step of a staircase. They can, and must, jump exactly
𝑘
k steps downward at a time. Can they successfully reach the 0-th step?

For example, if the child starts at the n=9-th step and can jump k=3 steps at a time,
the sequence of steps will be 9→6→3→0 successfully reaching the 0-th step.
"""

while True:
    try:
        n, k = map(int, input().split())
    except EOFError:
        break
    if n == 0:
        print("Ok!")
        continue

    if k == 0:
        print("Impossib1e!")
        continue

    if n % k == 0:
        print("Ok!")
    else:
        print("Impossib1e!")