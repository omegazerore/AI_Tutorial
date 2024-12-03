"""
輸入一行包含一個整數，且不超過
"""

    n = input()

    if n == '0':
        print(n)
    else:
        n = n[::-1].lstrip("0")
        if len(n) == 0:
            print("0")
        else:
            print(n)