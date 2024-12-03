"""
還記得計算機概論嗎？還記得二進位嗎？

現在我們來計算一下將一個10進位的數字換成二進位數字
"""

def decimal_to_binary(number):

    if number//2 == 0:
        return str(number % 2)
    else:
        return decimal_to_binary(number//2) + str(number % 2)


while True:
    try:
        number = int(input())
    except EOFError:
        break

    print(decimal_to_binary(number))
