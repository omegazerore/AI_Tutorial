'''
AC 34ms 3.3MB
'''

a, b = list(map(int, input().split()))

n = int(input())

person = 0

for _ in range(n):
    shopping_record = list(map(int, input().split()))
    a_list = []
    b_list = []
    for record in shopping_record:
        if abs(record) == a:
            a_list.append(record)
        elif abs(record) == b:
            b_list.append(record)
        else:
            continue
    if ((sum(a_list) > 0) and (sum(b_list) > 0)):
        person += 1

print(person)