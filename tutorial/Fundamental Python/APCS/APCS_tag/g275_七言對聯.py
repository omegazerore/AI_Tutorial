'''
AC 29ms 3.4MB
'''

n = int(input())

for _ in range(n):
    violation = ''
    sentence_binary_1 = list(map(int, input().split()))
    sentence_binary_2 = list(map(int, input().split()))
    # condition_1:
    if (sentence_binary_1[1] != sentence_binary_1[5]) or (sentence_binary_1[1] == sentence_binary_1[3]) or (sentence_binary_2[1] != sentence_binary_2[5]) or (sentence_binary_2[1] == sentence_binary_2[3]):
        violation += 'A'


    # condition_2:
    if (sentence_binary_1[-1] != 1) or (sentence_binary_2[-1] != 0):
        violation += 'B'

    # condition_3:
    if (sentence_binary_1[1] == sentence_binary_2[1]) or (sentence_binary_1[3] == sentence_binary_2[3]) or (sentence_binary_1[5] == sentence_binary_2[5]):
        violation += 'C'

    if violation == '':
        print("None")
    else:
        print(violation)