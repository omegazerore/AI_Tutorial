"""
明明是一個愛數（ㄕㄨˇ）數（ㄕㄨˋ）的好學生，這天媽媽叫他從 n 開始數，下一個數字是 n+1，再下一個數字是 n+2，以此類推。媽媽想知道，明明數了幾個數字之後，他數過的這些數字的總和會超過 m。請幫助明明的媽媽吧。
"""
import math

while True:
    try:
        n, m = list(map(int, input().split()))
    except EOFError:
        break

    b = (2 * n - 1)

    k = (-b + (b**2 + 8 * m)**0.5)/2

    print(math.ceil(k))