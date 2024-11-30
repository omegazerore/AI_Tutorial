'''
問題描述

將一個十進位正整數的奇數位數的和稱為A ，偶數位數的和稱為B，則A與B的絕對差值 |A －B| 稱為這個正整數的秘密差。

例如： 263541 的奇數位和 A = 6+5+1 =12，偶數位的和 B = 2+3+4 = 9 ，所以 263541 的秘密差是 |12 －9|= 3 。

給定一個 十進位正整數 X，請找出 X的秘密差。

AC 32ms 3.3MB
'''

n = str(input())
odd_summation = sum([int(t) for t in n[::2]])
even_summation = sum([int(t) for t in n[1::2]])

print(abs(even_summation - odd_summation))

