"""
AC 21ms 3.3MB
"""

_ = input()

elements = []

while True:
    try:
        elements.append(max(list(map(int, input().split()))))
    except EOFError:
        break

S = sum(elements)
print(S)

divisions = [element for element in elements if S%element==0]
if len(divisions) > 0:
    print(*divisions)
else:
    print(-1)
